#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Run the white-box TTK kernel CSV module without LLM orchestration.

This wrapper intentionally contains only workflow orchestration. CSV field
mapping, validation, and environment precheck remain delegated to the existing
fixed scripts so the normal path can run as a single command.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


REPORT_FILE = "ttk_module_report.json"
PRECHECK_FILE = "ttk_precheck_report.json"


@dataclass
class KernelRunCase:
    csv_path: Path
    result_path: Path
    case_name: str
    log_path: Path


@dataclass
class StepConfig:
    label: str
    fail_reason: str
    fail_msg: str


@dataclass
class KernelExecEnv:
    ops_test_kit_path: Path
    setenv_path: str | None


@dataclass
class CsvGenPaths:
    generate_script: Path
    operator_model: Path
    op_def_cpp: Path
    low_cases: Path
    high_cases: Path
    low_csv: Path
    high_csv: Path


@dataclass
class PrecheckPaths:
    precheck_script: Path
    whitebox_dir: Path
    ops_test_kit_path: Path
    golden_path: Path
    precheck_report: Path


def now_s() -> float:
    return time.perf_counter()


def command_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def run_command(command: list[str], cwd: Path | None = None, log_path: Path | None = None) -> dict[str, Any]:
    start = now_s()
    try:
        if log_path is None:
            proc = subprocess.run(
                command,
                cwd=str(cwd) if cwd else None,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            stdout = proc.stdout
            stderr = proc.stderr
        else:
            with log_path.open("w", encoding="utf-8") as log_file:
                proc = subprocess.run(
                    command,
                    cwd=str(cwd) if cwd else None,
                    text=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            stdout = ""
            stderr = ""
        return {
            "cmd": command_string(command),
            "cwd": str(cwd) if cwd else os.getcwd(),
            "returncode": proc.returncode,
            "duration_s": round(now_s() - start, 3),
            "stdout_tail": tail(stdout),
            "stderr_tail": tail(stderr),
            "log_path": str(log_path) if log_path else None,
        }
    except Exception as exc:  # noqa: BLE001 - surfaced in report
        return {
            "cmd": command_string(command),
            "cwd": str(cwd) if cwd else os.getcwd(),
            "returncode": None,
            "duration_s": round(now_s() - start, 3),
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "log_path": str(log_path) if log_path else None,
        }


def run_shell(command: str, cwd: Path | None = None, log_path: Path | None = None) -> dict[str, Any]:
    result = run_command(
        ["bash", "-lc", command],  # noqa: G.EDV.04 - shell required for source builtins
        cwd=cwd,
        log_path=log_path,
    )
    result["cmd"] = command
    return result


def fail_step(step: dict[str, Any], reason: str) -> dict[str, Any]:
    step["status"] = "failed"
    step["reason"] = reason
    return step


def pass_step(step: dict[str, Any]) -> dict[str, Any]:
    step["status"] = "passed"
    return step


def skip_step(step: dict[str, Any], reason: str) -> dict[str, Any]:
    step["status"] = "skipped"
    step["reason"] = reason
    return step


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_cases(path: Path) -> int:
    data = load_json(path)
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return len(data["cases"])
    raise RuntimeError(f"{path} is not a case list")


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)


def pick_case_name(csv_path: Path) -> str:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"{csv_path} contains no testcase rows")
    names = [row.get("testcase_name", "") for row in rows]
    if "case00001" in names:
        return "case00001"
    for name in names:
        if name and "empty" not in name.lower():
            return name
    return names[len(names) // 2]


def read_log(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _build_result_warnings(checks: dict[str, Any]) -> list[str]:
    warnings = []
    if not checks["perf_pass"]:
        warnings.append("perf_status_not_pass")
    if not checks["precision_pass"]:
        warnings.append("precision_status_not_pass")
    if not checks["memory_oob_pass"]:
        warnings.append("memory_oob_status_not_pass")
    if not checks["loaded_custom_golden"]:
        warnings.append("custom_golden_not_loaded")
    if not checks["status_pass_in_log"]:
        warnings.append("status_log_not_pass")
    if not checks["precision_pass_in_log"]:
        warnings.append("precision_log_not_pass")
    return warnings


def _read_csv_row(result_csv: Path) -> dict[str, str] | None:
    with result_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def _build_perf_checks(row: dict[str, str]) -> dict[str, Any]:
    perf_status = row.get("perf_status", "")
    precision_status = row.get("precision_status", "")
    memory_oob_status = row.get("memory_oob_status", "")
    return {
        "perf_status": perf_status,
        "precision_status": precision_status,
        "memory_oob_status": memory_oob_status,
        "perf_pass": perf_status == "PASS",
        "precision_pass": precision_status == "PASS",
        "memory_oob_pass": memory_oob_status in ("", "PASS"),
    }


def _build_log_checks(log_text: str, op_name: str) -> dict[str, Any]:
    return {
        "loaded_custom_golden": f"Loaded custom golden: kernel.{op_name}" in log_text,
        "compile_success": "Compilation Result: SUCC" in log_text,
        "status_pass_in_log": "STATUS: PASS" in log_text,
        "precision_pass_in_log": "PRECISION_STATUS: PASS" in log_text,
    }


def inspect_result(result_csv: Path, log_path: Path, op_name: str) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "result_csv_exists": result_csv.is_file(),
        "log_exists": log_path.is_file(),
    }
    if not result_csv.is_file():
        return {
            "status": "failed",
            "execution_pass": False,
            "checks": checks,
            "warnings": [],
            "reason": f"missing result csv: {result_csv}",
        }

    row = _read_csv_row(result_csv)
    if row is None:
        return {
            "status": "failed",
            "execution_pass": False,
            "checks": checks,
            "warnings": [],
            "reason": f"empty result csv: {result_csv}",
        }

    checks.update(_build_perf_checks(row))
    log_text = read_log(log_path)
    checks.update(_build_log_checks(log_text, op_name))
    warnings = _build_result_warnings(checks)

    execution_pass = (
        checks["result_csv_exists"]
        and checks["log_exists"]
        and checks["compile_success"]
    )
    if not execution_pass:
        status = "failed"
    elif warnings:
        status = "passed_with_warnings"
    else:
        status = "passed"
    return {
        "status": status,
        "execution_pass": execution_pass,
        "checks": checks,
        "warnings": warnings,
        "row": row,
    }


def build_ttk_command(args: argparse.Namespace, csv_path: Path, result_path: Path, case_name: str) -> list[str]:
    return [
        "python3",
        "-m",
        "ttk",
        "kernel",
        "-i",
        str(csv_path),
        "-o",
        str(result_path),
        "--plugin",
        str((Path(args.op_path).resolve() / "tests" / "assets" / "golden.py")),
        "-t",
        case_name,
        "--pc",
        "1",
        "--seed",
        "42",
    ]


def shell_ttk_command(command: list[str], ops_test_kit_path: Path, setenv_path: str) -> str:
    return (
        f"source {shlex.quote(setenv_path)} && "
        f"cd {shlex.quote(str(ops_test_kit_path))} && "
        f"{command_string(command)}"
    )


def run_ttk_kernel(
    args: argparse.Namespace,
    case: KernelRunCase,
    ops_test_kit_path: Path,
    setenv_path: str | None,
) -> dict[str, Any]:
    command = build_ttk_command(args, case.csv_path, case.result_path, case.case_name)
    if setenv_path:
        return run_shell(shell_ttk_command(command, ops_test_kit_path, setenv_path), log_path=case.log_path)
    return run_command(command, cwd=ops_test_kit_path, log_path=case.log_path)


def build_input_check(
    required_for_csv: list[Path],
    optional_for_kernel: list[Path],
) -> dict[str, Any]:
    missing_required = [str(path) for path in required_for_csv if not path.exists()]
    missing_optional = [str(path) for path in optional_for_kernel if not path.exists()]
    return {
        "status": "passed" if not missing_required else "failed",
        "required_for_csv": [str(path) for path in required_for_csv],
        "optional_for_kernel": [str(path) for path in optional_for_kernel],
        "missing_required_for_csv": missing_required,
        "missing_optional_for_kernel": missing_optional,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run white-box TTK kernel module as a fixed script workflow.")
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--whitebox-dir", required=True)
    parser.add_argument("--op-path", required=True)
    parser.add_argument("--op-def-cpp", required=True)
    parser.add_argument("--ops-test-kit-path", required=True)
    parser.add_argument("--skill-base", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--setenv-path", default=None)
    parser.add_argument("--skip-kernel-run", action="store_true")
    return parser.parse_args()


def _run_single_kernel(
    report: dict[str, Any],
    args: argparse.Namespace,
    case: KernelRunCase,
    env: KernelExecEnv,
    config: StepConfig,
) -> None:
    step = run_ttk_kernel(args, case, env.ops_test_kit_path, env.setenv_path)
    step["testcase_name"] = case.case_name
    step["inspection"] = inspect_result(case.result_path, case.log_path, args.op_name)
    if step["inspection"]["status"] == "failed":
        report["steps"][config.label] = fail_step(step, "kernel_execution_failed")
    elif step["inspection"]["status"] == "passed_with_warnings":
        report["steps"][config.label] = skip_step(step, "kernel_execution_passed_with_warnings")
        report["steps"][config.label]["status"] = "passed_with_warnings"
        for warning in step["inspection"].get("warnings", []):
            report["warnings"].append({"step": config.label, "reason": warning})
    else:
        report["steps"][config.label] = pass_step(step)
    if report["steps"][config.label]["status"] not in ("passed", "passed_with_warnings"):
        report["issues"].append({"step": config.label, "reason": config.fail_reason})
        raise RuntimeError(config.fail_msg)


def _run_csv_generation(
    report: dict[str, Any],
    args: argparse.Namespace,
    paths: CsvGenPaths,
) -> None:
    generate_cmd = [
        "python3", str(paths.generate_script),
        "--op-name", args.op_name,
        "--operator-model", str(paths.operator_model),
        "--op-def-cpp", str(paths.op_def_cpp),
        "--low-cases", str(paths.low_cases),
        "--high-cases", str(paths.high_cases),
        "--low-csv", str(paths.low_csv),
        "--high-csv", str(paths.high_csv),
    ]
    step = run_command(generate_cmd)
    if step["returncode"] != 0:
        report["steps"]["generate_csv"] = fail_step(step, "command_failed")
        report["issues"].append({"step": "generate_csv", "reason": "command_failed"})
        raise RuntimeError("generate_csv failed")
    low_rows = count_csv_rows(paths.low_csv)
    high_rows = count_csv_rows(paths.high_csv)
    step["low_csv_rows"] = low_rows
    step["high_csv_rows"] = high_rows
    report["counts"]["low_csv_rows"] = low_rows
    report["counts"]["high_csv_rows"] = high_rows
    if low_rows != report["counts"]["low_cases"] or high_rows != report["counts"]["high_cases"]:
        report["steps"]["generate_csv"] = fail_step(step, "csv_row_count_mismatch")
        report["issues"].append(
            {
                "step": "generate_csv",
                "reason": "csv_row_count_mismatch",
                "low_cases": report["counts"]["low_cases"],
                "low_csv_rows": low_rows,
                "high_cases": report["counts"]["high_cases"],
                "high_csv_rows": high_rows,
            }
        )
        raise RuntimeError("csv row count mismatch")
    report["steps"]["generate_csv"] = pass_step(step)


def _run_csv_validation(report, validate_script, low_csv, high_csv):
    for label, csv_path in (("validate_low_csv", low_csv), ("validate_high_csv", high_csv)):
        validate_cmd = ["python3", str(validate_script), str(csv_path)]
        step = run_command(validate_cmd)
        report["steps"][label] = pass_step(step) if step["returncode"] == 0 else fail_step(step, "command_failed")
        if step["returncode"] != 0:
            report["issues"].append({"step": label, "reason": "csv_validation_failed"})
            raise RuntimeError(f"{label} failed")


def _run_precheck(
    report: dict[str, Any],
    args: argparse.Namespace,
    paths: PrecheckPaths,
) -> dict[str, Any]:
    precheck_cmd = [
        "python3", str(paths.precheck_script),
        "--whitebox-dir", str(paths.whitebox_dir),
        "--ops-test-kit-path", str(paths.ops_test_kit_path),
        "--golden-path", str(paths.golden_path),
        "--op-name", args.op_name,
    ]
    if args.setenv_path:
        precheck_cmd.extend(["--setenv-path", args.setenv_path])
    step = run_command(precheck_cmd)
    if not paths.precheck_report.is_file():
        report["steps"]["precheck"] = fail_step(step, "missing_precheck_report")
        report["issues"].append({"step": "precheck", "reason": "precheck_report_missing"})
        raise RuntimeError("precheck report missing")
    precheck = load_json(paths.precheck_report)
    report["kernel_gate"] = precheck.get("kernel_gate", {"status": "unknown", "reason": "missing_kernel_gate"})
    step["precheck_status"] = precheck.get("status")
    step["kernel_gate"] = report["kernel_gate"]
    if step["returncode"] != 0 and precheck.get("status") is None:
        report["steps"]["precheck"] = fail_step(step, "command_failed")
        report["issues"].append({"step": "precheck", "reason": "command_failed"})
        raise RuntimeError("precheck command failed")
    if report["kernel_gate"].get("status") == "passed":
        report["steps"]["precheck"] = pass_step(step)
    else:
        report["steps"]["precheck"] = skip_step(
            step, report["kernel_gate"].get("reason", "kernel_gate_not_passed")
        )
    return precheck


def _build_paths(args: argparse.Namespace, skill_base: Path) -> dict[str, Path]:
    scripts_dir = skill_base / "scripts"
    whitebox_dir = Path(args.whitebox_dir).resolve()
    op_path = Path(args.op_path).resolve()
    ops_test_kit_path = Path(args.ops_test_kit_path).resolve()
    op_def_cpp = Path(args.op_def_cpp).resolve()
    return {
        "whitebox_dir": whitebox_dir,
        "op_path": op_path,
        "ops_test_kit_path": ops_test_kit_path,
        "op_def_cpp": op_def_cpp,
        "scripts_dir": scripts_dir,
        "low_cases": whitebox_dir / "S5_cases_low.json",
        "high_cases": whitebox_dir / "S5_cases_high.json",
        "operator_model": whitebox_dir / "S2P1_operator_model.json",
        "generate_script": scripts_dir / "ttk_generate_kernel_csv.py",
        "validate_script": scripts_dir / "ttk_validate_csv.py",
        "precheck_script": scripts_dir / "ttk_precheck_env.py",
        "low_csv": whitebox_dir / f"ttk_{args.op_name}_cases_low.csv",
        "high_csv": whitebox_dir / f"ttk_{args.op_name}_cases_high.csv",
        "precheck_report": whitebox_dir / PRECHECK_FILE,
        "low_result": whitebox_dir / f"ttk_{args.op_name}_cases_low_one_result.csv",
        "high_result": whitebox_dir / f"ttk_{args.op_name}_cases_high_one_result.csv",
        "low_log": whitebox_dir / "ttk_low_one.log",
        "high_log": whitebox_dir / "ttk_high_one.log",
        "golden_path": op_path / "tests" / "assets" / "golden.py",
        "report_path": whitebox_dir / REPORT_FILE,
    }


def _run_kernels(
    report: dict[str, Any],
    args: argparse.Namespace,
    paths: dict[str, Path],
    precheck: dict[str, Any],
) -> None:
    cann_env = precheck.get("checks", {}).get("cann_env", {})
    setenv_path = cann_env.get("setenv_path") if cann_env.get("status") == "passed_after_source" else None
    exec_env = KernelExecEnv(ops_test_kit_path=paths["ops_test_kit_path"], setenv_path=setenv_path)

    low_case_name = pick_case_name(paths["low_csv"])
    low_case = KernelRunCase(
        csv_path=paths["low_csv"], result_path=paths["low_result"],
        case_name=low_case_name, log_path=paths["low_log"],
    )
    _run_single_kernel(
        report, args, low_case, exec_env,
        StepConfig("run_low_one", "low_kernel_execution_failed", "low kernel execution failed"),
    )

    high_case_name = pick_case_name(paths["high_csv"])
    high_case = KernelRunCase(
        csv_path=paths["high_csv"], result_path=paths["high_result"],
        case_name=high_case_name, log_path=paths["high_log"],
    )
    _run_single_kernel(
        report, args, high_case, exec_env,
        StepConfig("run_high_one", "high_kernel_execution_failed", "high kernel execution failed"),
    )


def _run_input_check(
    report: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    required_for_csv = [
        paths["whitebox_dir"], paths["low_cases"], paths["high_cases"],
        paths["operator_model"], paths["op_def_cpp"],
        paths["generate_script"], paths["validate_script"], paths["precheck_script"],
    ]
    optional_for_kernel = [paths["ops_test_kit_path"], paths["golden_path"]]
    input_step = build_input_check(required_for_csv, optional_for_kernel)
    report["steps"]["input_check"] = input_step
    if input_step["status"] != "passed":
        report["issues"].append(
            {
                "step": "input_check",
                "reason": "missing_required_for_csv",
                "paths": input_step["missing_required_for_csv"],
            }
        )
        raise RuntimeError("missing required inputs for CSV generation")
    report["counts"]["low_cases"] = count_cases(paths["low_cases"])
    report["counts"]["high_cases"] = count_cases(paths["high_cases"])


def _run_workflow(
    report: dict[str, Any],
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> None:
    _run_input_check(report, paths)

    _run_csv_generation(
        report, args,
        CsvGenPaths(
            generate_script=paths["generate_script"],
            operator_model=paths["operator_model"],
            op_def_cpp=paths["op_def_cpp"],
            low_cases=paths["low_cases"],
            high_cases=paths["high_cases"],
            low_csv=paths["low_csv"],
            high_csv=paths["high_csv"],
        ),
    )
    _run_csv_validation(report, paths["validate_script"], paths["low_csv"], paths["high_csv"])
    precheck = _run_precheck(
        report, args,
        PrecheckPaths(
            precheck_script=paths["precheck_script"],
            whitebox_dir=paths["whitebox_dir"],
            ops_test_kit_path=paths["ops_test_kit_path"],
            golden_path=paths["golden_path"],
            precheck_report=paths["precheck_report"],
        ),
    )

    if args.skip_kernel_run:
        report["status"] = "skipped"
        report["issues"].append({"step": "kernel_run", "reason": "skip_kernel_run_requested"})
        return

    if report["kernel_gate"].get("status") != "passed":
        report["status"] = "skipped"
        report["issues"].append(
            {
                "step": "kernel_run",
                "status": "skipped",
                "reason": report["kernel_gate"].get("reason", "kernel_gate_not_passed"),
                "detail": "CSV files were generated and validated, but TTK kernel validation was skipped.",
                "preserved_outputs": ["low_csv", "high_csv", "precheck_report"],
            }
        )
        return

    _run_kernels(report, args, paths, precheck)
    report["status"] = "passed_with_warnings" if report["warnings"] else "passed"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    start = now_s()
    skill_base = Path(args.skill_base).resolve()
    paths = _build_paths(args, skill_base)

    report: dict[str, Any] = {
        "status": "failed",
        "op_name": args.op_name,
        "duration_s": 0.0,
        "counts": {},
        "steps": {},
        "outputs": {
            "low_csv": str(paths["low_csv"]),
            "high_csv": str(paths["high_csv"]),
            "precheck_report": str(paths["precheck_report"]),
            "low_result_csv": str(paths["low_result"]),
            "high_result_csv": str(paths["high_result"]),
            "low_log": str(paths["low_log"]),
            "high_log": str(paths["high_log"]),
            "module_report": str(paths["report_path"]),
        },
        "kernel_gate": {"status": "unknown", "reason": "not_run"},
        "issues": [],
        "warnings": [],
    }

    try:
        _run_workflow(report, args, paths)
    except Exception as exc:  # noqa: BLE001 - keep a machine-readable report for diagnosis
        report["issues"].append({"step": "module", "reason": str(exc)})
    finally:
        report["duration_s"] = round(now_s() - start, 3)
        write_report(paths["report_path"], report)
        _logger.info(f"TTK module status: {report['status']}")
        _logger.info(f"TTK module report: {paths['report_path']}")
        if report["status"] != "passed":
            raise SystemExit(1 if report["status"] == "failed" else 0)


if __name__ == "__main__":
    main()
