#!/usr/bin/env python3
"""Validate Markdown contracts and maintain CATLASS development state."""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path


TITLE = "# CATLASS DSL Workflow Contract"
SECTIONS = (
    "Approval",
    "Repository Identity",
    "Operator Specification",
    "Tensor Interface",
    "Semantic Requirements",
    "Preliminary Design",
    "Scope And Allowed Paths",
    "Approved Commands",
    "Required Cases",
    "Performance Target",
    "Risk Classification",
    "Evidence And Delivery",
)
STATUSES = {"passed", "failed", "blocked", "not_run"}
PHASES = {"configuration", "build", "focused_test", "full_test", "benchmark", "profiling"}
CATEGORIES = {"correctness", "boundary", "safety", "performance"}
SAFE_ID = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
RUN_ID = re.compile(
    r"^[a-z0-9]+(?:[._-][a-z0-9]+)*-(?P<timestamp>\d{8}-\d{6})$"
)
APPROVAL_FIELDS = (
    "revision", "approval_status", "approved_by", "approved_at", "contract_digest"
)
REPOSITORY_FIELDS = ("repository_root",)
SPECIFICATION_FIELDS = ("operator_name", "operator_family", "purpose")
SEMANTIC_FIELDS = ("computation", "boundary_behavior", "numerical_behavior")
PRELIMINARY_DESIGN_FIELDS = (
    "algorithm", "dataflow", "tiling_and_layout", "memory_and_sync"
)
PERFORMANCE_FIELDS = (
    "required",
    "metric_path",
    "direction",
    "threshold",
    "benchmark_command_id",
    "max_iterations",
    "stall_threshold",
    "min_improvement_fraction",
    "profiling_required",
)
RISK_FIELDS = ("risk_level", "risk_reasons", "targeted_review_focus")
DELIVERY_FIELDS = (
    "design_document", "design_sha256", "evidence_directory",
    "knowledge_mode", "delivery",
)
OUTPUT_FIELDS = (
    "status", "config_digest", "tasks", "tests", "review",
    "benchmark", "knowledge", "evidence", "next_action",
)
ANTI_HACK_POLICY = "single-fused-catlass-kernel-v1"


class ContractError(ValueError):
    pass


def _validate_anti_hack(value, label, single_observed=False):
    if not isinstance(value, dict) or value.get("status") != "passed":
        raise ContractError("{} anti_hack.status=passed 是必需的".format(label))
    if value.get("policy") != ANTI_HACK_POLICY:
        raise ContractError("{} anti_hack.policy 非法".format(label))
    declared = value.get("declared_kernel_names")
    observed = value.get("observed_kernel_names")
    if (
        not isinstance(declared, list)
        or not declared
        or any(not isinstance(item, str) or not item for item in declared)
        or len(set(declared)) != len(declared)
    ):
        raise ContractError("{} declared_kernel_names 非法".format(label))
    if (
        not isinstance(observed, list)
        or not observed
        or any(not isinstance(item, str) or not item for item in observed)
        or not set(observed).issubset(declared)
        or (single_observed and len(observed) != 1)
    ):
        raise ContractError("{} observed_kernel_names 非法".format(label))
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
        raise ContractError("{} anti_hack launch 证据非法".format(label))
    return value


def _profile_kernel_evidence(trace_root, anti_hack, label):
    trace_root = Path(trace_root)
    direct = trace_root / "kernel_details.csv"
    if not direct.is_file() or direct.is_symlink():
        raise ContractError("{} 必须有合并 kernel_details.csv".format(label))
    names = []
    try:
        with direct.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"Name", "Type", "OP State"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ContractError("{} kernel_details 缺少来源字段".format(label))
            for row in reader:
                name = (row.get("Name") or "").strip()
                if (
                    not name
                    or (row.get("Type") or "").strip() != name
                    or (row.get("OP State") or "").strip() not in {"", "N/A"}
                ):
                    raise ContractError("{} kernel_details 非 CATLASS 证据".format(label))
                names.append(name)
    except OSError as exc:
        raise ContractError("{} kernel_details 不可读".format(label)) from exc
    if (
        len(names) != anti_hack["profiled_iterations"]
        or set(names) != set(anti_hack["observed_kernel_names"])
    ):
        raise ContractError("{} kernel_details 与 anti_hack 不一致".format(label))
    manifest_path = trace_root / "anti_hack_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ContractError("{} 缺少 anti_hack_manifest.json".format(label))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("{} anti_hack manifest 非法".format(label)) from exc
    entries = manifest.get("iterations") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("policy") != ANTI_HACK_POLICY
        or manifest.get("profiled_iterations") != anti_hack["profiled_iterations"]
        or not isinstance(entries, list)
        or len(entries) != anti_hack["profiled_iterations"]
    ):
        raise ContractError("{} anti_hack manifest 与结果不一致".format(label))
    iteration_files = []
    iteration_names = []
    for expected, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("iteration") != expected:
            raise ContractError("{} iteration 顺序非法".format(label))
        relative = entry.get("kernel_details")
        if not isinstance(relative, str) or not relative:
            raise ContractError("{} iteration kernel_details 非法".format(label))
        expected_relative = Path("anti_hack") / "iteration-{:04d}".format(expected) / "kernel_details.csv"
        if Path(relative) != expected_relative:
            raise ContractError("{} iteration kernel_details 路径非 canonical".format(label))
        unresolved = trace_root / relative
        relative_parts = Path(relative).parts
        if any(
            (trace_root.joinpath(*relative_parts[:index])).is_symlink()
            for index in range(1, len(relative_parts) + 1)
        ):
            raise ContractError("{} iteration 证据不能是符号链接".format(label))
        path = unresolved.resolve()
        try:
            path.relative_to(trace_root.resolve())
        except ValueError as exc:
            raise ContractError("{} iteration 证据越界".format(label)) from exc
        if path.is_symlink() or not path.is_file():
            raise ContractError("{} iteration 证据缺失".format(label))
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256"):
            raise ContractError("{} iteration 证据哈希不一致".format(label))
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ContractError("{} 每个 iteration 必须恰好一次 launch".format(label))
        row = rows[0]
        name = (row.get("Name") or "").strip()
        if (
            not name
            or (row.get("Type") or "").strip() != name
            or (row.get("OP State") or "").strip() not in {"", "N/A"}
            or entry.get("observed_launches") != 1
            or entry.get("observed_kernel_names") != [name]
        ):
            raise ContractError("{} iteration 非 CATLASS 单 launch 证据".format(label))
        iteration_files.append(path)
        iteration_names.append(name)
    if set(iteration_names) != set(anti_hack["observed_kernel_names"]):
        raise ContractError("{} iteration 名称与 anti_hack 不一致".format(label))
    return {
        "kernel_details": direct,
        "manifest": manifest_path,
        "iterations": iteration_files,
    }


def _strip_code(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == "`" and value[-1] == "`":
        return value[1:-1]
    return value


def _parse_table(body, section):
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if len(lines) < 2 or not lines[0].startswith("|") or not lines[1].startswith("|"):
        raise ContractError("{} 必须包含 Markdown 表格".format(section))

    def cells(line):
        return [_strip_code(cell) for cell in line.strip("|").split("|")]

    headers = cells(lines[0])
    separator = cells(lines[1])
    if len(headers) != len(separator) or any(not re.fullmatch(r":?-{3,}:?", cell) for cell in separator):
        raise ContractError("{} 表头分隔行非法".format(section))
    rows = []
    for line in lines[2:]:
        if not line.startswith("|"):
            raise ContractError("{} 表格外存在非空内容".format(section))
        values = cells(line)
        if len(values) != len(headers):
            raise ContractError("{} 表格列数不一致".format(section))
        rows.append(dict(zip(headers, values)))
    if not rows:
        raise ContractError("{} 表格不能为空".format(section))
    return headers, rows


def _field_table(tables, section, expected):
    headers, rows = tables[section]
    if headers != ["field", "value"]:
        raise ContractError("{} 必须使用 field/value 列".format(section))
    result = {}
    for row in rows:
        key = row["field"]
        if key in result:
            raise ContractError("{} 包含重复字段 {}".format(section, key))
        result[key] = row["value"]
    if tuple(result) != expected:
        raise ContractError("{} 字段或顺序非法".format(section))
    return result


def parse_contract(path):
    path = Path(path)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError("契约必须是 UTF-8") from exc
    if not text.startswith(TITLE + "\n"):
        raise ContractError("契约标题非法")
    matches = list(re.finditer(r"^## ([^\n]+)\n", text, re.MULTILINE))
    names = tuple(match.group(1) for match in matches)
    if names != SECTIONS:
        raise ContractError("契约章节缺失、重复或顺序非法")
    tables = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        tables[match.group(1)] = _parse_table(text[match.end():end], match.group(1))

    contract = {
        "path": str(path.resolve()),
        "raw": raw,
        "approval": _field_table(tables, "Approval", APPROVAL_FIELDS),
        "repository": _field_table(tables, "Repository Identity", REPOSITORY_FIELDS),
        "specification": _field_table(
            tables, "Operator Specification", SPECIFICATION_FIELDS
        ),
        "semantics": _field_table(
            tables, "Semantic Requirements", SEMANTIC_FIELDS
        ),
        "preliminary_design": _field_table(
            tables, "Preliminary Design", PRELIMINARY_DESIGN_FIELDS
        ),
        "performance": _field_table(tables, "Performance Target", PERFORMANCE_FIELDS),
        "risk": _field_table(tables, "Risk Classification", RISK_FIELDS),
        "delivery": _field_table(tables, "Evidence And Delivery", DELIVERY_FIELDS),
    }
    interface_headers, contract["tensor_interface"] = tables["Tensor Interface"]
    scope_headers, contract["allowed_paths"] = tables["Scope And Allowed Paths"]
    command_headers, contract["commands"] = tables["Approved Commands"]
    case_headers, contract["required_cases"] = tables["Required Cases"]
    if interface_headers != [
        "name", "direction", "shape", "dtype", "layout", "semantics"
    ]:
        raise ContractError("Tensor Interface 列非法")
    if scope_headers != ["path", "purpose"]:
        raise ContractError("Scope And Allowed Paths 列非法")
    if command_headers != ["command_id", "phase", "argv"]:
        raise ContractError("Approved Commands 列非法")
    if case_headers != ["case_id", "category", "inputs", "oracle", "command_id"]:
        raise ContractError("Required Cases 列非法")
    return contract


def contract_digest(raw):
    text = raw.decode("utf-8")
    pattern = re.compile(
        r"(?m)^(\|\s*contract_digest\s*\|\s*)(?:`?[0-9a-f]{64}`?|`?<computed>`?)(\s*\|)$"
    )
    normalized, count = pattern.subn(r"\1<computed>\2", text)
    if count != 1:
        raise ContractError("Approval 必须包含唯一 contract_digest")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def config_digest(config):
    value = {key: item for key, item in config.items() if key not in {"raw", "path"}}
    approval = dict(value.get("approval", {}))
    approval.pop("contract_digest", None)
    approval["config_digest"] = "<computed>"
    value["approval"] = approval
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _safe_relative(value, label):
    path = Path(value)
    if (
        not value or path.is_absolute() or value in {".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ContractError("{} 必须是安全仓库相对路径".format(label))
    return value


def validate_contract(contract, require_approved=False):
    errors = []
    approval = contract["approval"]
    try:
        revision = int(approval["revision"])
        if revision < 1:
            raise ValueError
    except ValueError:
        errors.append("revision 必须是正整数")
    if approval["approval_status"] not in {"draft", "approved"}:
        errors.append("approval_status 只能是 draft 或 approved")
    digest = contract_digest(contract["raw"]) if "raw" in contract else config_digest(contract)
    if approval["approval_status"] == "approved":
        if not approval["approved_by"] or not approval["approved_at"]:
            errors.append("approved 契约必须填写 approved_by 和 approved_at")
        try:
            datetime.datetime.fromisoformat(approval["approved_at"].replace("Z", "+00:00"))
        except ValueError:
            errors.append("approved_at 必须是 RFC-3339 时间")
        expected_digest = approval.get("config_digest", approval.get("contract_digest"))
        if expected_digest != digest:
            errors.append("config_digest 与配置内容不匹配")
    elif require_approved:
        errors.append("契约尚未批准")

    for section, values in (
        ("Operator Specification", contract["specification"]),
        ("Semantic Requirements", contract["semantics"]),
        ("Preliminary Design", contract["preliminary_design"]),
    ):
        for field, value in values.items():
            if not value.strip():
                errors.append("{} 的 {} 不能为空".format(section, field))

    seen_tensors = set()
    directions = set()
    for row in contract["tensor_interface"]:
        name = row["name"]
        if not SAFE_ID.fullmatch(name) or name in seen_tensors:
            errors.append("Tensor Interface name 非法或重复：{}".format(name))
        seen_tensors.add(name)
        if row["direction"] not in {"input", "output"}:
            errors.append("Tensor Interface direction 只能是 input 或 output")
        else:
            directions.add(row["direction"])
        for field in ("shape", "dtype", "layout", "semantics"):
            if not row[field].strip():
                errors.append("Tensor Interface {} 不能为空：{}".format(field, name))
    if directions != {"input", "output"}:
        errors.append("Tensor Interface 必须同时声明 input 和 output")

    seen_paths = set()
    for row in contract["allowed_paths"]:
        try:
            value = _safe_relative(row["path"], "allowed path")
            if value in seen_paths:
                errors.append("allowed path 重复：{}".format(value))
            seen_paths.add(value)
        except ContractError as exc:
            errors.append(str(exc))
        if not row["purpose"]:
            errors.append("allowed path purpose 不能为空")
    try:
        kernel_value = contract.get("kernel_path")
        if kernel_value is None and len(seen_paths) == 1:
            kernel_value = next(iter(seen_paths))
        kernel_path = _safe_relative(kernel_value, "kernel_path")
        if kernel_path not in seen_paths:
            errors.append("kernel_path 必须引用 allowed_paths 中的文件")
    except (ContractError, TypeError) as exc:
        errors.append(str(exc))

    commands = {}
    for row in contract["commands"]:
        command_id = row["command_id"]
        if not SAFE_ID.fullmatch(command_id) or command_id in commands:
            errors.append("command_id 非法或重复：{}".format(command_id))
        if row["phase"] not in PHASES:
            errors.append("command phase 非法：{}".format(row["phase"]))
        try:
            argv = json.loads(row["argv"])
            if not isinstance(argv, list) or not argv or any(
                not isinstance(item, str) or not item for item in argv
            ):
                raise ValueError
            row["parsed_argv"] = argv
        except (json.JSONDecodeError, ValueError):
            errors.append("command argv 必须是非空 JSON 字符串数组：{}".format(command_id))
        commands[command_id] = row

    seen_cases = set()
    for row in contract["required_cases"]:
        case_id = row["case_id"]
        if not SAFE_ID.fullmatch(case_id) or case_id in seen_cases:
            errors.append("case_id 非法或重复：{}".format(case_id))
        seen_cases.add(case_id)
        if row["category"] not in CATEGORIES:
            errors.append("required case category 非法：{}".format(case_id))
        if not row["inputs"] or not row["oracle"]:
            errors.append("required case inputs/oracle 不能为空：{}".format(case_id))
        if row["command_id"] not in commands:
            errors.append("required case 引用未知 command_id：{}".format(case_id))
        elif (
            row["category"] == "performance"
            and commands[row["command_id"]]["phase"] != "benchmark"
        ):
            errors.append("performance case 必须引用 benchmark command")

    performance = contract["performance"]
    if performance["required"] not in {"yes", "no"}:
        errors.append("Performance Target required 只能是 yes 或 no")
    command_id = performance["benchmark_command_id"]
    if command_id not in commands or commands[command_id]["phase"] != "benchmark":
        errors.append("最终性能基线必须引用 benchmark phase command")
    if not re.fullmatch(
        r"[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+",
        performance["metric_path"],
    ):
        errors.append("metric_path 必须是点分隔的 benchmark JSON 数值路径")
    if performance["required"] == "yes":
        if performance["profiling_required"] not in {"yes", "no"}:
            errors.append("profiling_required 只能是 yes 或 no")
        if not any(
            row["category"] != "performance"
            for row in contract["required_cases"]
        ):
            errors.append("性能任务必须包含至少一个非 performance required case")
        if not any(
            row["category"] == "performance"
            and row["command_id"] == command_id
            for row in contract["required_cases"]
        ):
            errors.append("性能任务必须包含引用 benchmark_command_id 的 performance case")
        if performance["direction"] not in {"lower", "higher"}:
            errors.append("direction 只能是 lower 或 higher")
        try:
            threshold = float(performance["threshold"])
            if not threshold > 0 or not threshold < float("inf"):
                raise ValueError
        except ValueError:
            errors.append("threshold 必须是有限正数")
        try:
            max_iterations = int(performance["max_iterations"])
            stall_threshold = int(performance["stall_threshold"])
            if not 1 <= max_iterations <= 100:
                raise ValueError
            if not 1 <= stall_threshold <= max_iterations:
                raise ValueError
        except ValueError:
            errors.append("max_iterations 必须为 1..100，stall_threshold 必须位于其范围内")
        try:
            improvement = float(performance["min_improvement_fraction"])
            if not 0 <= improvement < 1:
                raise ValueError
        except ValueError:
            errors.append("min_improvement_fraction 必须位于 [0, 1)")
    else:
        if performance["profiling_required"] != "yes":
            errors.append("无性能门槛时 profiling_required 必须为 yes")
        for key in (
            "direction", "threshold", "max_iterations", "stall_threshold",
            "min_improvement_fraction",
        ):
            if performance[key] != "not_applicable":
                errors.append("无性能门槛时 {} 必须是 not_applicable".format(key))

    risk = contract["risk"]
    if risk["risk_level"] not in {"standard", "high"}:
        errors.append("risk_level 只能是 standard 或 high")
    if risk["risk_level"] == "high" and risk["targeted_review_focus"] in {"", "not_applicable"}:
        errors.append("high risk 必须填写 targeted_review_focus")
    if risk["risk_level"] == "standard" and risk["targeted_review_focus"] != "not_applicable":
        errors.append("standard risk 的 targeted_review_focus 必须是 not_applicable")

    delivery = contract["delivery"]
    try:
        design_document = _safe_relative(
            delivery["design_document"], "design_document"
        )
        design_root = Path(contract["design_root"]).resolve() if "design_root" in contract else (Path(contract["path"]).resolve().parent if "path" in contract else Path(contract["repository"]["repository_root"]))
        design_path = design_root / design_document
        if (
            not design_path.is_file()
            or design_path.is_symlink()
            or design_path.name != "DESIGN.md"
        ):
            errors.append("design_document 必须指向 repository_root 下的 DESIGN.md 普通文件")
        else:
            actual_design_digest = hashlib.sha256(design_path.read_bytes()).hexdigest()
            if delivery["design_sha256"] != actual_design_digest:
                errors.append("design_sha256 与 DESIGN.md 内容不匹配")
    except ContractError as exc:
        errors.append(str(exc))
    if not re.fullmatch(r"[0-9a-f]{64}", delivery["design_sha256"]):
        errors.append("design_sha256 必须是 64 位小写 SHA-256")
    try:
        evidence_directory = _safe_relative(
            delivery["evidence_directory"], "evidence_directory"
        )
        evidence_parts = Path(evidence_directory).parts
        run_match = (
            RUN_ID.fullmatch(evidence_parts[2])
            if len(evidence_parts) == 3
            else None
        )
        if (
            len(evidence_parts) != 3
            or evidence_parts[0] != ".catlass-dsl"
            or evidence_parts[1] not in {"develop-runs", "optimize-runs"}
            or run_match is None
        ):
            errors.append(
                "evidence_directory run-id 必须以 YYYYMMDD-HHMMSS 时间戳结尾"
            )
        else:
            try:
                datetime.datetime.strptime(
                    run_match.group("timestamp"), "%Y%m%d-%H%M%S"
                )
            except ValueError:
                errors.append("evidence_directory run-id 时间戳不是有效的 UTC 日期时间")
    except ContractError as exc:
        errors.append(str(exc))
    if delivery["knowledge_mode"] != "batch_at_finish":
        errors.append("knowledge_mode 必须是 batch_at_finish")
    if delivery["delivery"] != "working_tree":
        errors.append("delivery 必须是 working_tree")
    if errors:
        raise ContractError("; ".join(dict.fromkeys(errors)))
    return digest


def verify_workspace(contract):
    repository = contract["repository"]
    root = Path(repository["repository_root"])
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ContractError("repository_root 必须是无符号链接的绝对目录")
    if str(root.resolve()) != str(root):
        raise ContractError("repository_root 必须是 canonical path")
    return root


def _empty_output(digest, next_action):
    return {
        "status": "not_run",
        "config_digest": digest,
        "tasks": [],
        "tests": {},
        "review": {},
        "benchmark": {},
        "knowledge": {"queries": [], "admission": {}},
        "evidence": [],
        "next_action": next_action,
    }


def start_state(contract):
    return start_state_at(contract, "task_breakdown")


def start_state_at(contract, entry_stage):
    if "kernel_path" not in contract and len(contract.get("allowed_paths", [])) == 1:
        contract["kernel_path"] = contract["allowed_paths"][0]["path"]
    digest = validate_contract(contract, require_approved=True)
    verify_workspace(contract)
    if entry_stage not in {
        "task_breakdown", "implement", "debug", "final_review", "benchmark", "optimize"
    }:
        raise ContractError("entry_stage 非法")
    if (
        entry_stage == "optimize"
        and contract["performance"]["required"] != "yes"
    ):
        raise ContractError("optimize 独立入口要求已批准性能目标")
    config = {key: value for key, value in contract.items() if key not in {"raw", "path"}}
    config["design_root"] = contract.get("design_root", str(Path(contract["path"]).resolve().parent) if "path" in contract else contract["repository"]["repository_root"])
    config["approval"] = dict(config["approval"])
    config["approval"].pop("contract_digest", None)
    config["approval"]["config_digest"] = config_digest(config)
    digest = config["approval"]["config_digest"]
    state = _empty_output(digest, entry_stage)
    state.update({"schema": "catlass.dsl.workflow.v3", "revision": int(config["approval"]["revision"]), "config": config, "events": []})
    return state


def _next_stage(contract, stage, result, events):
    passed = result["status"] == "passed"
    if not passed:
        if stage == "task_breakdown":
            return (
                "blocked"
                if result["status"] in {"blocked", "not_run"}
                else "task_breakdown"
            )
        if stage == "optimize":
            return "blocked"
        return "blocked" if result["status"] in {"blocked", "not_run"} else "implement"
    if stage == "task_breakdown":
        return "implement"
    if stage == "implement":
        return (
            "targeted_review"
            if contract["risk"]["risk_level"] == "high"
            else "final_review"
        )
    if stage == "debug":
        return (
            "targeted_review"
            if contract["risk"]["risk_level"] == "high"
            else "final_review"
        )
    if stage == "targeted_review":
        return "final_review"
    if stage == "full_test":
        return "benchmark"
    if stage == "final_review":
        last_optimize = max(
            (index for index, event in enumerate(events) if event["stage"] == "optimize"),
            default=-1,
        )
        last_benchmark = max(
            (index for index, event in enumerate(events) if event["stage"] == "benchmark"),
            default=-1,
        )
        return "finish" if last_optimize >= 0 and last_benchmark > last_optimize else "full_test"
    if stage == "benchmark":
        if result.get("correctness") != "passed":
            return "implement"
        if (
            contract["performance"]["required"] == "yes"
            and result.get("performance_target_met") is not True
        ):
            return "optimize"
        return (
            "final_review"
            if any(event["stage"] == "optimize" for event in events)
            else "finish"
        )
    if stage == "optimize":
        return "benchmark"
    if stage == "finish":
        return "complete"
    raise ContractError("未知 stage：{}".format(stage))


def _validate_task_breakdown(contract, result):
    tasks = result.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ContractError("task_breakdown 必须包含非空 tasks")
    allowed_paths = {row["path"] for row in contract["allowed_paths"]}
    required_cases = {
        row["case_id"]
        for row in contract["required_cases"]
        if row["category"] != "performance"
    }
    seen = set()
    covered_paths = set()
    covered_cases = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {
            "task_id", "depends_on", "allowed_paths", "required_cases", "done_when"
        }:
            raise ContractError("task 字段集合非法")
        task_id = task["task_id"]
        if not isinstance(task_id, str) or not SAFE_ID.fullmatch(task_id) or task_id in seen:
            raise ContractError("task_id 非法或重复")
        dependencies = task["depends_on"]
        if (
            not isinstance(dependencies, list)
            or any(not isinstance(item, str) for item in dependencies)
            or len(dependencies) != len(set(dependencies))
            or any(item not in seen for item in dependencies)
        ):
            raise ContractError("depends_on 必须只引用前序 task")
        paths = task["allowed_paths"]
        if (
            not isinstance(paths, list)
            or not paths
            or any(not isinstance(item, str) for item in paths)
            or any(item not in allowed_paths for item in paths)
        ):
            raise ContractError("task allowed_paths 必须引用契约批准路径")
        cases = task["required_cases"]
        if (
            not isinstance(cases, list)
            or not cases
            or any(not isinstance(item, str) for item in cases)
            or any(item not in required_cases for item in cases)
        ):
            raise ContractError("task required_cases 必须引用契约 required cases")
        if not isinstance(task["done_when"], str) or not task["done_when"].strip():
            raise ContractError("task done_when 不能为空")
        covered_paths.update(paths)
        covered_cases.update(cases)
        seen.add(task_id)
    if covered_paths != allowed_paths:
        raise ContractError("task breakdown 必须覆盖全部契约批准路径")
    if covered_cases != required_cases:
        raise ContractError("task breakdown 必须覆盖全部非 performance required cases")
    return tasks


def _validate_benchmark_result(contract, result):
    if result.get("correctness") != "passed":
        raise ContractError("benchmark passed 必须包含 correctness=passed")
    environment = result.get("environment", {})
    device = environment.get("device") if isinstance(environment, dict) else None
    is_npu = isinstance(device, str) and device.split(":", 1)[0] == "npu"
    if is_npu:
        _validate_anti_hack(result.get("anti_hack"), "NPU benchmark")
    workloads = result.get("workloads")
    if (
        not isinstance(workloads, list)
        or not workloads
        or any(not isinstance(item, dict) for item in workloads)
    ):
        raise ContractError("NPU benchmark workloads 非法")
    if is_npu:
        for index, workload in enumerate(workloads):
            _validate_anti_hack(
                workload.get("anti_hack"),
                "NPU workload {}".format(index),
                single_observed=True,
            )
    performance = result.get("performance")
    if not isinstance(performance, dict) or performance.get("status") != "passed":
        raise ContractError("benchmark passed 必须包含 performance.status=passed")
    candidate = performance.get("candidate")
    if not isinstance(candidate, dict):
        raise ContractError("benchmark performance 必须包含 candidate")
    mean_ms = candidate.get("mean_ms")
    if (
        not isinstance(mean_ms, (int, float))
        or isinstance(mean_ms, bool)
        or not math.isfinite(mean_ms)
        or not mean_ms > 0
    ):
        raise ContractError("benchmark candidate.mean_ms 必须是有限正数")
    profiling = result.get("profiling")
    if not isinstance(profiling, dict) or profiling.get("status") != "passed":
        raise ContractError("benchmark passed 必须包含 profiling.status=passed")
    profiled_candidate = profiling.get("candidate")
    if (
        not isinstance(profiled_candidate, dict)
        or not isinstance(profiled_candidate.get("artifact"), str)
        or not profiled_candidate["artifact"].strip()
    ):
        raise ContractError("benchmark profiling 必须包含 candidate artifact")
    if (
        contract["performance"]["required"] == "yes"
        and not isinstance(result.get("performance_target_met"), bool)
    ):
        raise ContractError("有性能门槛时 benchmark 必须记录 performance_target_met")
    _workload_speedups(result, require_complete=True)


def _workload_speedups(benchmark, require_complete=False):
    workloads = benchmark.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        if require_complete:
            raise ContractError("benchmark 必须包含逐 workload 性能结果")
        return []
    records = []
    seen = set()
    for workload in workloads:
        uuid = workload.get("uuid") if isinstance(workload, dict) else None
        performance = workload.get("performance") if isinstance(workload, dict) else None
        candidate = performance.get("candidate") if isinstance(performance, dict) else None
        reference = performance.get("reference") if isinstance(performance, dict) else None
        candidate_ms = candidate.get("mean_ms") if isinstance(candidate, dict) else None
        reference_ms = reference.get("mean_ms") if isinstance(reference, dict) else None

        def valid_number(value):
            return (
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value > 0
            )

        if (
            not isinstance(uuid, str) or not uuid.strip() or uuid in seen
            or workload.get("status") != "passed"
            or not isinstance(performance, dict) or performance.get("status") != "passed"
            or not valid_number(candidate_ms) or not valid_number(reference_ms)
        ):
            if require_complete:
                raise ContractError("benchmark workload 性能结果不完整或非法")
            continue
        seen.add(uuid)
        records.append({
            "uuid": uuid,
            "candidate_mean_ms": float(candidate_ms),
            "reference_mean_ms": float(reference_ms),
            "speedup": float(reference_ms) / float(candidate_ms),
        })
    return records


def _validate_finish(state):
    if state.get("tests", {}).get("status") != "passed":
        raise ContractError("finish 要求当前 full_test passed")
    if state.get("review", {}).get("final_review", {}).get("status") != "passed":
        raise ContractError("finish 要求当前 final_review passed")
    if state.get("benchmark", {}).get("benchmark", {}).get("status") != "passed":
        raise ContractError("finish 要求当前 benchmark passed")
    benchmark = state.get("benchmark", {}).get("benchmark", {})
    environment = benchmark.get("environment", {})
    device = environment.get("device") if isinstance(environment, dict) else None
    if not isinstance(device, str) or device.split(":", 1)[0] != "npu":
        raise ContractError("finish 要求当前 benchmark 来自 NPU")
    _validate_anti_hack(benchmark.get("anti_hack"), "finish NPU benchmark")


def _validate_knowledge_queries(result, kernel_sha256):
    queries = result.get("knowledge_queries", [])
    if not isinstance(queries, list):
        raise ContractError("knowledge_queries 必须是列表")
    required = {
        "trigger", "failure_signature", "filters", "matches",
        "evidence", "kernel_sha256",
    }
    optional = {"retrieval", "match_details"}
    validated = []
    for query in queries:
        if (
            not isinstance(query, dict)
            or not required.issubset(query)
            or set(query) - required - optional
        ):
            raise ContractError("knowledge query 字段集合非法")
        if not all(
            isinstance(query[field], str) and query[field].strip()
            for field in ("trigger", "failure_signature", "evidence")
        ):
            raise ContractError("knowledge query 标识和证据不能为空")
        if query["kernel_sha256"] != kernel_sha256:
            raise ContractError("knowledge query kernel_sha256 与当前 kernel 不一致")
        filters = query["filters"]
        if not isinstance(filters, dict) or not filters:
            raise ContractError("knowledge query filters 必须是非空对象")
        allowed_filters = {"type", "tags", "status", "operator_family", "arch", "text"}
        if any(key not in allowed_filters for key in filters):
            raise ContractError("knowledge query filter 非法")
        tags = filters.get("tags", [])
        if (
            not isinstance(tags, list)
            or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
        ):
            raise ContractError("knowledge query tags 必须是字符串列表")
        if any(
            key != "tags" and (not isinstance(value, str) or not value.strip())
            for key, value in filters.items()
        ):
            raise ContractError("knowledge query filter 值不能为空")
        matches = query["matches"]
        if (
            not isinstance(matches, list)
            or any(
                not isinstance(path, str)
                or not path.startswith(".catlass-dsl/knowledge/")
                or not path.endswith(".md")
                or ".." in Path(path).parts
                for path in matches
            )
        ):
            raise ContractError(
                "knowledge query matches 必须是安全的 .catlass-dsl/knowledge Markdown 路径"
            )
        retrieval = query.get("retrieval")
        if retrieval is not None:
            if (
                not isinstance(retrieval, dict)
                or set(retrieval) != {
                    "normalized_text", "normalized_operator_family", "match_mode"
                }
                or retrieval.get("match_mode") not in {
                    "exact", "all_terms", "relaxed"
                }
                or any(
                    value is not None and (
                        not isinstance(value, str) or not value.strip()
                    )
                    for key, value in retrieval.items() if key != "match_mode"
                )
            ):
                raise ContractError("knowledge query retrieval 非法")
        details = query.get("match_details")
        if details is not None:
            allowed_fields = {
                "title", "operator_families", "tags", "description", "path",
                "body", "source_titles",
            }
            if not isinstance(details, list):
                raise ContractError("knowledge query match_details 必须是列表")
            for detail in details:
                if (
                    not isinstance(detail, dict)
                    or set(detail) != {
                        "path", "score", "matched_fields", "matched_terms"
                    }
                    or detail.get("path") not in matches
                    or not isinstance(detail.get("score"), int)
                    or isinstance(detail.get("score"), bool)
                    or detail["score"] < 0
                    or not isinstance(detail.get("matched_fields"), list)
                    or any(field not in allowed_fields for field in detail["matched_fields"])
                    or not isinstance(detail.get("matched_terms"), list)
                    or any(
                        not isinstance(term, str) or not term.strip()
                        for term in detail["matched_terms"]
                    )
                ):
                    raise ContractError("knowledge query match_details 非法")
        validated.append(query)
    return validated


def _normalize_knowledge_state(state):
    knowledge = state.get("knowledge")
    if knowledge == {}:
        knowledge = {"queries": [], "admission": {}}
        state["knowledge"] = knowledge
    if (
        not isinstance(knowledge, dict)
        or set(knowledge) != {"queries", "admission"}
        or not isinstance(knowledge["queries"], list)
        or not isinstance(knowledge["admission"], dict)
    ):
        raise ContractError("state knowledge 结构非法")
    return knowledge


def advance_state(contract, state, result):
    contract = state.get("config", contract)
    digest = validate_contract(contract, require_approved=True)
    if state.get("schema") != "catlass.dsl.workflow.v3":
        raise ContractError("state schema 非法")
    if state.get("config_digest") != digest:
        raise ContractError("契约变化，旧状态已失效")
    expected = state.get("next_action")
    if expected in {"blocked", "complete"}:
        raise ContractError("当前状态不能继续推进")
    if result.get("stage") != expected:
        raise ContractError("result stage 与 next_action 不一致")
    if result.get("status") not in STATUSES:
        raise ContractError("result status 非法")
    if not isinstance(result.get("evidence"), list):
        raise ContractError("result evidence 必须是列表")
    knowledge = _normalize_knowledge_state(state)
    verify_workspace(contract)
    kernel = Path(contract["repository"]["repository_root"]) / contract["kernel_path"]
    kernel_sha256 = hashlib.sha256(kernel.read_bytes()).hexdigest() if kernel.is_file() and not kernel.is_symlink() else None
    if result.get("kernel_sha256") != kernel_sha256:
        raise ContractError("result kernel_sha256 与当前 kernel 不一致")
    if expected == "task_breakdown" and result["status"] == "passed":
        state["tasks"] = _validate_task_breakdown(contract, result)
    if expected == "benchmark" and result["status"] == "passed":
        _validate_benchmark_result(contract, result)
    if expected == "finish" and result["status"] == "passed":
        _validate_finish(state)
    if "knowledge_queries" in result and expected not in {"implement", "debug"}:
        raise ContractError("knowledge_queries 只允许用于 implement/debug")
    knowledge_queries = _validate_knowledge_queries(result, kernel_sha256)
    seen_queries = {
        (query["kernel_sha256"], query["failure_signature"])
        for query in knowledge["queries"]
    }
    new_queries = [
        (query["kernel_sha256"], query["failure_signature"])
        for query in knowledge_queries
    ]
    if len(new_queries) != len(set(new_queries)) or any(
        key in seen_queries for key in new_queries
    ):
        raise ContractError("同一 kernel 和 failure_signature 不得重复查询")

    event = dict(result)
    event["kernel_sha256"] = kernel_sha256
    event["attempt"] = 1 + sum(item["stage"] == expected for item in state["events"])
    state["events"].append(event)
    state["status"] = result["status"]
    state["evidence"].extend(result["evidence"])
    if result.get("changed"):
        state["tests"] = {}
        state["review"] = {}
        state["benchmark"] = {}
        knowledge["admission"] = {}
    knowledge["queries"].extend(knowledge_queries)
    if expected == "full_test":
        state["tests"] = result
    elif expected in {"targeted_review", "final_review"}:
        state["review"][expected] = result
    elif expected in {"benchmark", "optimize"}:
        state["benchmark"][expected] = result
    elif expected == "finish":
        knowledge["admission"] = result.get("knowledge", {})
    state["next_action"] = _next_stage(contract, expected, result, state["events"])
    if state["next_action"] == "complete":
        state["status"] = "passed"
    return state


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".state.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".document.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _run_root(contract):
    relative = Path(contract["delivery"]["evidence_directory"])
    if relative.parts[:2] != (".catlass-dsl", "develop-runs"):
        raise ContractError("develop 控制器只接受 .catlass-dsl/develop-runs/<run-id>")
    return (Path(contract["repository"]["repository_root"]) / relative).resolve()


def _stage_dir(run_root, sequence, stage):
    return Path(run_root) / "traces" / "iter-{:03d}-{}".format(sequence, stage)


def _relative_to_run(path, run_root):
    return str(Path(path).resolve().relative_to(Path(run_root).resolve()))


def _review_evidence(contract, result, stage_dir):
    project = Path(contract["repository"]["repository_root"])
    run_root = _run_root(contract)
    if result["status"] == "passed" and not result["evidence"]:
        raise ContractError("passed stage 必须提供至少一条可审查 evidence")
    records = []
    for value in result["evidence"]:
        if not isinstance(value, str):
            raise ContractError("result evidence 必须是项目相对路径字符串列表")
        relative = _safe_relative(value, "result evidence")
        candidate = project / relative
        resolved = candidate.resolve()
        try:
            resolved.relative_to(stage_dir.resolve())
        except ValueError:
            raise ContractError("result evidence 必须位于当前 trace 目录")
        if not candidate.is_file() or candidate.is_symlink():
            raise ContractError("result evidence 必须是无符号链接的普通文件")
        records.append(
            {
                "path": _relative_to_run(candidate, run_root),
                "kind": "file", "size": candidate.stat().st_size,
                "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            }
        )
    supplied = set(result["evidence"])
    if any(
        query.get("evidence") not in supplied
        for query in result.get("knowledge_queries", [])
    ):
        raise ContractError("knowledge query evidence 必须包含在当前 stage evidence 中")
    return records


def _md(value):
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _iterations_markdown(contract, state):
    lines = [
        "# Iteration Log", "", "## Summary", "",
        "| Iter | Title | Score | Passed | Notes |",
        "|------|-------|-------|--------|-------|",
    ]
    baseline = None
    for event in state["events"]:
        performance = event.get("performance", {}).get("candidate", {})
        mean = performance.get("mean_ms")
        if event["stage"] == "benchmark" and isinstance(mean, (int, float)):
            baseline = baseline or float(mean)
        score = "—" if not mean or not baseline else "{:.4f}x".format(baseline / float(mean))
        note = event.get("summary") or event.get("decision") or "next={}".format(event.get("next_action", state["next_action"]))
        lines.append("| {} | {} | {} | {} | {} |".format(
            event["sequence"], _md(event["stage"]), score,
            "yes" if event["status"] == "passed" else "no", _md(note),
        ))
    lines.extend([
        "",
        "## Notes",
        "",
        "- Operator: {}".format(_md(contract["specification"]["operator_name"])),
        "- Status: {}; next action: {}".format(_md(state["status"]), _md(state["next_action"])),
    ])
    for event in state["events"]:
        if event["stage"] == "debug":
            lines.append("- Iter {} Expected: {}".format(event["sequence"], _md(event.get("expected", event.get("summary", "single falsifiable debug hypothesis")))))
    if state.get("next_action") == "complete" and state["events"]:
        final = state["events"][-1]
        lines.append("- Final kernel SHA-256: `{}`".format(final.get("kernel_sha256") or "unavailable"))
        lines.append("- Knowledge admission: {}".format(_md(state.get("knowledge", {}).get("admission") or "none")))
        benchmark = next((
            event for event in reversed(state["events"])
            if event.get("stage") == "benchmark" and event.get("status") == "passed"
        ), None)
        workload_speedups = _workload_speedups(benchmark or {})
        if workload_speedups:
            lines.extend([
                "", "## Workload Speedups", "",
                "| Workload | Candidate (ms) | Reference (ms) | Speedup |",
                "|----------|---------------:|---------------:|--------:|",
            ])
            for workload in workload_speedups:
                lines.append("| {} | {:.6f} | {:.6f} | {:.4f}x |".format(
                    _md(workload["uuid"]), workload["candidate_mean_ms"],
                    workload["reference_mean_ms"], workload["speedup"],
                ))
    return "\n".join(lines) + "\n"


def _write_iterations(contract, state):
    _atomic_text(_run_root(contract) / "ITERATIONS.md", _iterations_markdown(contract, state))


def _prune(root, allowed):
    allowed = {Path(item).resolve() for item in allowed}
    for child in list(Path(root).iterdir()):
        resolved = child.resolve()
        if any(resolved == item or str(item).startswith(str(resolved) + os.sep) for item in allowed):
            continue
        shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()


def _compact_profile(result, destination, project):
    artifact = result.get("profiling", {}).get("candidate", {}).get("artifact")
    if not artifact:
        return []
    source = Path(artifact)
    source = Path(project) / source if not source.is_absolute() else source
    if source.is_symlink():
        raise ContractError("candidate profiler artifact 必须是 trace 内的目录")
    source = source.resolve()
    trace_root = Path(destination).parent.resolve()
    if not source.is_dir():
        raise ContractError("candidate profiler artifact 必须是 trace 内的目录")
    try:
        source.relative_to(trace_root)
    except ValueError as exc:
        raise ContractError("candidate profiler artifact 必须位于当前 trace 内") from exc
    copied = []
    environment = result.get("environment", {})
    device = environment.get("device") if isinstance(environment, dict) else None
    is_npu_benchmark = (
        result.get("stage") == "benchmark"
        and isinstance(device, str)
        and device.split(":", 1)[0] == "npu"
    )
    if is_npu_benchmark:
        for index, workload in enumerate(result.get("workloads", [])):
            timing = workload.get("performance", {}).get("candidate", {})
            trace = timing.get("trace_dir")
            if not trace:
                raise ContractError(
                    "NPU workload {} 缺少 candidate trace_dir".format(index)
                )
            workload_source = Path(trace)
            workload_source = (
                Path(project) / workload_source
                if not workload_source.is_absolute()
                else workload_source
            )
            if workload_source.is_symlink() or not workload_source.is_dir():
                raise ContractError("candidate workload profiler 必须是普通目录")
            workload_source = workload_source.resolve()
            try:
                workload_source.relative_to(source)
            except ValueError as exc:
                raise ContractError(
                    "candidate workload profiler 必须位于 artifact 内"
                ) from exc
            anti_hack = _validate_anti_hack(
                workload.get("anti_hack"),
                "NPU workload {}".format(index),
                single_observed=True,
            )
            kernel_evidence = _profile_kernel_evidence(
                workload_source, anti_hack, "NPU workload {}".format(index)
            )
            case_destination = Path(destination) / "case-{:04d}".format(index)
            for name in ("kernel_details.csv", "step_trace_time.csv"):
                direct = workload_source / name
                matches = (
                    [direct]
                    if direct.is_file()
                    else sorted(workload_source.glob("**/{}".format(name)))
                )
                if name == "kernel_details.csv":
                    matches = [kernel_evidence["kernel_details"]]
                if matches:
                    target = case_destination / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(matches[0], target)
                    copied.append(target)
    else:
        for name in ("kernel_details.csv", "step_trace_time.csv"):
            direct = source / name
            matches = [direct] if direct.is_file() else sorted(
                source.glob("**/{}".format(name))
            )
            if matches:
                target = Path(destination) / "case-0000" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(matches[0], target)
                copied.append(target)
        if not any(path.name == "kernel_details.csv" for path in copied):
            raise ContractError("NPU anti-hack 要求保留 kernel_details.csv")
    if result.get("profiling", {}).get("status") == "passed" and not copied:
        raise ContractError("profiling 未生成 kernel_details.csv 或 step_trace_time.csv")
    return copied


def _initialize_run(contract, state_path, state):
    run_root = _run_root(contract)
    _require_state_path(contract, state_path)
    run_root.mkdir(parents=True, exist_ok=True)
    _assert_clean_run_root(run_root)
    (run_root / "traces").mkdir(exist_ok=True)
    _write_iterations(contract, state)


def _require_state_path(contract, state_path):
    expected = _run_root(contract) / "state.json"
    if Path(state_path).resolve() != expected:
        raise ContractError("state 必须位于 develop run 根目录的 state.json")


def _assert_clean_run_root(run_root):
    unexpected = sorted(
        path.name
        for path in run_root.iterdir()
        if path.name not in {"state.json", "ITERATIONS.md", "traces", "final"}
    )
    if unexpected:
        raise ContractError(
            "develop run 根目录包含未归档文件：{}".format(", ".join(unexpected))
        )


def _record_stage(contract, previous_state, result_path, result, state):
    run_root = _run_root(contract)
    _assert_clean_run_root(run_root)
    sequence = len(previous_state["events"]) + 1
    stage = previous_state["next_action"]
    stage_dir = _stage_dir(run_root, sequence, stage)
    expected_submission = stage_dir / "submission.json"
    if Path(result_path).resolve() != expected_submission:
        raise ContractError(
            "result 必须位于 traces/iter-NNN-<stage>/submission.json"
        )
    evidence_records = _review_evidence(contract, result, stage_dir)
    event = state["events"][-1]
    event["sequence"] = sequence
    event["next_action"] = state["next_action"]
    event["evidence"] = evidence_records
    state["evidence"] = [item for prior in state["events"] for item in prior.get("evidence", []) if isinstance(item, dict)]
    kernel = Path(contract["repository"]["repository_root"]) / contract["kernel_path"]
    if not kernel.is_file() or kernel.is_symlink():
        raise ContractError("iteration kernel 不存在或 identity 非法")
    if hashlib.sha256(kernel.read_bytes()).hexdigest() != event["kernel_sha256"]:
        raise ContractError("iteration 关闭前 kernel 已变化")
    kernel_snapshot = stage_dir / "kernel.py"
    shutil.copyfile(kernel, kernel_snapshot)
    profile_files = _compact_profile(result, stage_dir / "profile", contract["repository"]["repository_root"])
    _atomic_json(stage_dir / "result.json", event)
    allowed = [kernel_snapshot, stage_dir / "result.json", *profile_files]
    if result["status"] != "passed":
        failure = str(result.get("failure_summary") or result.get("reason") or result["status"])[:16384]
        _atomic_text(stage_dir / "failure.txt", failure + "\n")
        allowed.append(stage_dir / "failure.txt")
    _prune(stage_dir, allowed)
    if stage == "finish" and result["status"] == "passed":
        final = run_root / "final"
        final.mkdir(exist_ok=True)
        shutil.copyfile(kernel_snapshot, final / "kernel.py")
        final_result = {
            "kernel_sha256": event["kernel_sha256"],
            "tests": state["tests"], "review": state["review"],
            "benchmark": state["benchmark"], "knowledge": state["knowledge"],
            "status": "passed",
        }
        _atomic_json(final / "result.json", final_result)
        benchmark_event = next((item for item in reversed(state["events"]) if item["stage"] == "benchmark" and item["status"] == "passed"), None)
        profile_source_root = stage_dir / "profile"
        if benchmark_event:
            profile_source_root = _stage_dir(run_root, benchmark_event["sequence"], "benchmark") / "profile"
            profile_files = [
                path for path in profile_source_root.rglob("*") if path.is_file()
            ]
        final_profiles = []
        for source in profile_files:
            target = final / "profile" / source.relative_to(profile_source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            final_profiles.append(target)
        _prune(final, [final / "kernel.py", final / "result.json", *final_profiles])
    _write_iterations(contract, state)


def build_parser():
    parser = argparse.ArgumentParser(description="CATLASS DSL state controller")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--state", required=True)
    start = commands.add_parser("start")
    start.add_argument("--state", required=True)
    start.add_argument(
        "--entry-stage",
        choices=(
            "task_breakdown", "implement", "debug", "final_review",
            "benchmark", "optimize"
        ),
        default="task_breakdown",
    )
    advance = commands.add_parser("advance")
    advance.add_argument("--state", required=True)
    advance.add_argument("--result", required=True)
    status = commands.add_parser("status")
    status.add_argument("--state", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        state = json.loads(Path(args.state).read_text(encoding="utf-8"))
        contract = state.get("config")
        if not isinstance(contract, dict):
            raise ContractError("state.config 必须包含冻结的开发配置")
        if state.get("schema") != "catlass.dsl.workflow.v3":
            raise ContractError("state schema 非法")
        if args.command == "validate":
            digest = validate_contract(contract, require_approved=True)
            verify_workspace(contract)
            output = {"status": "passed", "config_digest": digest}
        elif args.command == "start":
            if contract.get("approval", {}).get("config_digest") == "<computed>":
                contract["approval"]["config_digest"] = config_digest(contract)
            output = start_state_at(contract, args.entry_stage)
            _initialize_run(contract, args.state, output)
            _atomic_json(args.state, output)
        elif args.command == "advance":
            _require_state_path(contract, args.state)
            result = json.loads(Path(args.result).read_text(encoding="utf-8"))
            previous = {
                "events": list(state.get("events", [])),
                "next_action": state.get("next_action"),
            }
            output = advance_state(contract, state, result)
            _record_stage(contract, previous, args.result, result, output)
            _atomic_json(args.state, output)
        else:
            validate_contract(contract, require_approved=True)
            output = state
    except (OSError, ContractError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
