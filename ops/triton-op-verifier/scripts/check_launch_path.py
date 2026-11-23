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

"""Check the declared artifacts for the single-launch delivery path."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


DELIVERY_PATH = "multi-kernel baseline -> fused epilogue -> single launch"


def _decorator_is_triton_jit(node: ast.AST) -> bool:
    """与 validate_triton_impl.py 保持一致：@triton.jit、@jit 及带参数调用形式。"""
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == "triton"
            and node.attr == "jit"
        )
    if isinstance(node, ast.Name):
        return node.id == "jit"
    if isinstance(node, ast.Call):
        return _decorator_is_triton_jit(node.func)
    return False


def read_artifact(op_dir: Path, name: str, *, process_evidence: bool) -> dict:
    path = (op_dir / name).resolve()
    if not path.is_relative_to(op_dir) or path.suffix != ".py":
        raise ValueError(
            f"expected a Python artifact inside the operator directory: {name}"
        )
    relative = path.relative_to(op_dir)
    if process_evidence != (relative.parts[0] == "dev"):
        raise ValueError(f"unexpected artifact location: {name}")

    source = path.read_bytes()
    tree = ast.parse(source, filename=str(relative))
    jit_count = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(_decorator_is_triton_jit(item) for item in node.decorator_list)
    )
    return {
        "path": relative.as_posix(),
        "jit_count": jit_count,
        "sha256": hashlib.sha256(source).hexdigest(),
    }


def validate_contract(op_dir: Path) -> None:
    precheck = json.loads((op_dir / "precheck.json").read_text(encoding="utf-8"))
    if not isinstance(precheck, dict):
        raise ValueError("precheck.json must contain an object")
    contract = precheck.get("delivery_contract", precheck)
    if not isinstance(contract, dict):
        raise ValueError("delivery_contract must contain an object")

    required_values = {
        "single_launch_required": True,
        "max_kernel_launches": 1,
        "delivery_path": DELIVERY_PATH,
    }
    for key, expected in required_values.items():
        value = contract.get(key)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"invalid delivery contract field: {key}")

    for key in ("golden_device", "golden_source"):
        value = contract.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing delivery contract field: {key}")

    config_path = op_dir.parent.parent / "triton_op_generator.config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("project config must contain an object")
        for key in (*required_values, "golden_device", "golden_source"):
            if key not in config:
                continue
            if (
                type(config[key]) is not type(contract[key])
                or config[key] != contract[key]
            ):
                raise ValueError(
                    f"delivery contract disagrees with project config: {key}"
                )

    anchor_path = op_dir / "output" / ".baseline_anchor.json"
    if anchor_path.is_file():
        anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
        if not isinstance(anchor, dict):
            raise ValueError("frozen baseline must contain an object")
        source_mode = anchor.get("source_mode")
        if source_mode in ("user", "auto") and source_mode != contract["golden_source"]:
            raise ValueError("golden_source disagrees with frozen baseline")

    for key in (*required_values, "golden_device", "golden_source"):
        if key in precheck and (
            type(precheck[key]) is not type(contract[key])
            or precheck[key] != contract[key]
        ):
            raise ValueError(f"conflicting delivery contract field: {key}")


def check_baseline_verify(op_dir, report_name, baselines):
    if not report_name:
        raise ValueError("missing --baseline-verify result")
    path = (op_dir / report_name).resolve()
    if not path.is_relative_to(op_dir):
        raise ValueError("baseline verification must belong to this operator run")
    report = json.loads(path.read_text(encoding="utf-8"))
    total = report.get("total_cases")
    passed = report.get("passed_cases")
    if (
        type(total) is not int
        or total <= 0
        or type(passed) is not int
        or passed != total
        or report.get("failures") != []
    ):
        raise ValueError(f"baseline verification incomplete: {passed}/{total}")
    sources = report.get("local_source_sha256", {})
    if not isinstance(sources, dict):
        raise ValueError("invalid verification source hashes")
    executed = report.get("implementation_sha256")
    if not isinstance(executed, str):
        raise ValueError("verification report has no executed entry hash")
    if all(item["sha256"] != executed for item in baselines):
        raise ValueError("declared baseline was not the executed verification entry")
    hashes = set(sources.values())
    hashes.add(executed)
    if any(item["sha256"] not in hashes for item in baselines):
        raise ValueError("verification does not cover the declared baseline source")
    return {
        "path": str(path.relative_to(op_dir)),
        "passed_cases": passed,
        "total_cases": total,
        "executed_baseline_sha256": executed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-dir", required=True, type=Path)
    parser.add_argument(
        "--single-launch-required", choices=("true", "false"), default="true"
    )
    parser.add_argument(
        "--phase",
        choices=("precheck", "iteration", "fusion", "final"),
        default="iteration",
    )
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument("--baseline-verify")
    parser.add_argument("--fused-epilogue")
    parser.add_argument("--final-code")
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Record an explicit multi-kernel fallback",
    )
    args = parser.parse_args()

    op_dir = args.op_dir.resolve()
    required = args.single_launch_required == "true"
    record = {
        "single_launch_required": required,
        "stage": args.phase,
        "op_dir": str(op_dir),
    }
    code = 0
    try:
        if not required:
            record["verdict"] = "not_required"
        else:
            validate_contract(op_dir)
            record["verdict"] = "pass_precheck"

        if required and args.phase != "precheck":
            baselines = [
                read_artifact(op_dir, name, process_evidence=True)
                for name in args.baseline
            ]
            record["baselines"] = baselines
            baseline_count = sum(
                artifact["jit_count"]
                for artifact in {item["path"]: item for item in baselines}.values()
            )
            if baseline_count < 2:
                raise ValueError("no declared multi-kernel baseline found")
            record["verdict"] = "pass_iteration"

        if required and args.phase in ("fusion", "final"):
            record["baseline_verify"] = check_baseline_verify(
                op_dir, args.baseline_verify, baselines
            )
            record["verdict"] = "pass_baseline_verify"

        if required and args.phase == "final":
            if not args.final_code or not args.fused_epilogue:
                raise ValueError(
                    "final phase requires --final-code and --fused-epilogue"
                )
            fused = read_artifact(op_dir, args.fused_epilogue, process_evidence=True)
            record["fused_epilogue"] = fused
            if not fused["jit_count"] or any(
                fused["sha256"] == item["sha256"] for item in baselines
            ):
                raise ValueError("fused epilogue must be a distinct kernel artifact")
            final = read_artifact(op_dir, args.final_code, process_evidence=False)
            record["final"] = final
            if not args.fallback and final["sha256"] == fused["sha256"]:
                raise ValueError(
                    "the final kernel is a copy of the fused epilogue stage"
                )
            if args.fallback and final["jit_count"] >= 2:
                record["verdict"] = "contract_conflict"
                code = 2
            elif args.fallback or final["jit_count"] != 1:
                raise ValueError(
                    "final artifact does not match the declared delivery form"
                )
            else:
                record["verdict"] = "pass_final_shape"
    except (OSError, ValueError, SyntaxError) as exc:
        record.update(verdict="delivery_path_violation", reason=str(exc))
        code = 1

    evidence = op_dir / "output" / "launch_path.json"
    try:
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"check-launch-path: cannot write evidence: {exc}")
        return 1

    print(f"check-launch-path: {record['verdict']}")
    if "reason" in record:
        print(record["reason"])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
