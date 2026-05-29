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
"""Review a CANNBot SKILL.md with repository gates plus a nine-dimension score."""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path


DIMENSIONS: list[tuple[str, int]] = [
    ("Frontmatter 质量", 8),
    ("工作流清晰度", 15),
    ("边界条件覆盖", 10),
    ("检查点设计", 7),
    ("指令具体性", 15),
    ("资源整合度", 5),
    ("CANNBot 架构适配性", 15),
    ("领域可信度与安全边界", 10),
    ("验证证据", 15),
]

OUTPUT_LOGGER = logging.getLogger("cannbot_skill_reviewer.output")
ERROR_LOGGER = logging.getLogger("cannbot_skill_reviewer.error")


def _configure_stream_logger(logger: logging.Logger, stream: object) -> None:
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _emit(message: str = "") -> None:
    OUTPUT_LOGGER.info("%s", message)


def _count_pattern(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE))


def _clamp(value: float) -> float:
    return float(max(1.0, min(10.0, round(value, 1))))


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for candidate in (cur, *cur.parents):
        if (candidate / "tests" / "lib" / "skill_validator.py").is_file():
            return candidate
    raise FileNotFoundError("cannot locate repository root with tests/lib/skill_validator.py")


def _resolve_skill_file(path_arg: str) -> Path:
    path = Path(path_arg).expanduser().resolve()
    if path.is_dir():
        path = path / "SKILL.md"
    if not path.is_file():
        raise FileNotFoundError(f"SKILL.md not found: {path}")
    return path


def _frontmatter_text(text: str) -> str:
    match = re.search(r"\A---\r?\n(?P<fm>.*?)\r?\n---\r?\n", text, flags=re.DOTALL)
    return match.group("fm") if match else ""


def _run_validator(repo_root: Path, skill_file: Path) -> list[dict[str, object]]:
    validator = repo_root / "tests" / "lib" / "skill_validator.py"
    cmd = [sys.executable, str(validator), "validate-skill", str(skill_file)]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=60,
    )

    findings: list[dict[str, object]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            findings.append({
                "level": "error",
                "rule": "VALIDATOR",
                "file": str(skill_file),
                "msg": f"validator emitted non-JSON output: {line}",
            })
            continue
        if "summary" not in item:
            findings.append(item)

    if proc.returncode != 0:
        findings.append({
            "level": "error",
            "rule": "VALIDATOR",
            "file": str(skill_file),
            "msg": proc.stderr.strip() or f"validator exited with {proc.returncode}",
        })
    return findings


def _count_features(text: str) -> dict[str, int]:
    return {
        "heading": _count_pattern(text, r"^##\s+"),
        "numbered_steps": _count_pattern(text, r"^\s*\d+\.\s+"),
        "workflow": _count_pattern(
            text,
            r"(工作流程|流程|步骤|Step|Phase|输入|输出|产物|交付)",
        ),
        "edge": _count_pattern(
            text,
            r"(异常|错误|失败|fallback|回退|降级|无法|缺失|不通过|blocked|recover|retry)",
        ),
        "checkpoint": _count_pattern(
            text,
            r"(确认|用户确认|AskUserQuestion|审批|暂停|人工|复核|review|写操作)",
        ),
        "code_block": _count_pattern(text, r"^```"),
        "command": _count_pattern(
            text,
            r"\b(python|bash|pytest|git|rg|npu-smi|curl|opencode|claude)\b",
        ),
        "path": _count_pattern(
            text,
            r"(\breferences/|\bscripts/|\bassets/|\btests/|\bdocs/|\.md\b|\.py\b)",
        ),
        "resource": _count_pattern(
            text,
            r"(references/|scripts/|assets/|templates?/|tests/|docs/)",
        ),
        "cann": _count_pattern(
            text,
            (
                r"(CANNBot|CANN|Ascend|NPU|torch_npu|ops/|model/|graph/|infra/|"
                r"STANDARDS|skill_validator|rules\.yaml)"
            ),
        ),
        "trust": _count_pattern(
            text,
            r"(可信|来源|官方|规范|禁止|不要|不得|只读|token|secret|密钥|凭据|网络|安全|风险)",
        ),
        "evidence": _count_pattern(
            text,
            (
                r"(测试|验证|自测|pytest|run-tests|skill_validator|validate-skill|"
                r"eval|Prompt|Expected|dry_run|NPU实测)"
            ),
        ),
    }


def _is_expected_layer(skill_file: Path, repo_root: Path) -> bool:
    rel_path = (
        skill_file.relative_to(repo_root)
        if skill_file.is_relative_to(repo_root)
        else skill_file
    )
    path_text = str(rel_path).replace("\\", "/")
    return any(
        path_text.startswith(prefix)
        for prefix in ("ops/", "ops-lab/", "model/", "graph/", "infra/")
    )


def _raw_scores(
    frontmatter: str,
    features: dict[str, int],
    in_expected_layer: bool,
    findings: list[dict[str, object]],
) -> list[float]:
    error_count = sum(1 for item in findings if item.get("level") == "error")
    warn_count = sum(1 for item in findings if item.get("level") == "warn")
    has_name = bool(re.search(r"(?m)^name:\s*\S+", frontmatter))
    has_desc = bool(re.search(r"(?m)^description:\s*.+", frontmatter))
    has_license = bool(re.search(r"(?m)^license:\s*.+", frontmatter))

    return [
        4.0
        + (2.0 if frontmatter else 0.0)
        + (1.0 if has_name else 0.0)
        + (1.5 if has_desc else 0.0)
        + (0.8 if has_license else 0.0) - min(3.0, error_count * 1.0),
        3.5
        + min(3.5, features["numbered_steps"] * 0.35)
        + min(1.5, features["heading"] * 0.2)
        + min(1.5, features["workflow"] * 0.2),
        3.0 + min(6.0, features["edge"] * 0.45),
        3.0 + min(6.0, features["checkpoint"] * 0.7),
        3.5
        + min(2.5, features["code_block"] * 0.35)
        + min(2.0, features["command"] * 0.3)
        + min(2.0, features["path"] * 0.15),
        3.5 + min(5.5, features["resource"] * 0.7),
        3.5
        + (2.0 if in_expected_layer else 0.0)
        + min(4.0, features["cann"] * 0.35)
        - min(1.5, warn_count * 0.15),
        3.5 + min(5.5, features["trust"] * 0.55),
        3.0 + min(6.0, features["evidence"] * 0.45),
    ]


def _score_dimensions(
    text: str,
    skill_file: Path,
    repo_root: Path,
    findings: list[dict[str, object]],
) -> list[dict[str, object]]:
    raw_scores = _raw_scores(
        _frontmatter_text(text),
        _count_features(text),
        _is_expected_layer(skill_file, repo_root),
        findings,
    )
    reasons = [
        "基于 frontmatter、name、description、license 与结构错误扣分。",
        "基于章节、编号步骤、输入输出与流程词覆盖度估算。",
        "基于异常、失败、fallback、降级和恢复路径描述估算。",
        "基于用户确认、人工复核、写操作前检查点描述估算。",
        "基于命令、代码块、路径、参数和输出格式的具体程度估算。",
        "基于 references/scripts/assets/tests/docs 等资源引用估算。",
        "基于目录分层、CANNBot/CANN/Ascend/NPU 及仓库规范适配度估算。",
        "基于可信来源、安全边界、凭据/网络/写操作约束描述估算。",
        "基于测试、验证、自测、门禁命令、dry_run 或 NPU 实测证据估算。",
    ]

    scored: list[dict[str, object]] = []
    for (name, weight), raw, reason in zip(DIMENSIONS, raw_scores, reasons):
        scored.append({
            "name": name,
            "weight": weight,
            "score": _clamp(raw),
            "reason": reason,
        })
    return scored


def _total_score(dimensions: list[dict[str, object]]) -> float:
    weighted = sum(item["score"] * item["weight"] for item in dimensions)
    return round(weighted / 10.0, 1)


def _verdict(
    total: float,
    findings: list[dict[str, object]],
    dimensions: list[dict[str, object]],
) -> str:
    has_error = any(item.get("level") == "error" for item in findings)
    low_key_dimension = any(item["score"] < 6.0 for item in dimensions if item["weight"] >= 10)
    if has_error or total < 70.0:
        return "REJECT"
    if total >= 80.0 and not low_key_dimension:
        return "PASS"
    return "CONDITIONAL"


def _suggestions(
    dimensions: list[dict[str, object]],
    findings: list[dict[str, object]],
) -> list[str]:
    tips: list[str] = []
    for item in findings:
        if item.get("level") == "error":
            tips.append(f"修复阻塞规则 {item.get('rule')}: {item.get('msg')}")
    for item in sorted(dimensions, key=lambda x: x["score"])[:3]:
        if item["score"] < 8.0:
            tips.append(f"优先补强「{item['name']}」（当前 {item['score']}/10）。")
    if not tips:
        tips.append("保留当前结构，并在 PR 描述中附上本报告和已运行的测试命令。")
    return tips


def _build_report(skill_file: Path, repo_root: Path) -> dict[str, object]:
    text = skill_file.read_text(encoding="utf-8", errors="replace")
    findings = _run_validator(repo_root, skill_file)
    dimensions = _score_dimensions(text, skill_file, repo_root, findings)
    total = _total_score(dimensions)
    verdict = _verdict(total, findings, dimensions)
    rel_path = (
        str(skill_file.relative_to(repo_root))
        if skill_file.is_relative_to(repo_root)
        else str(skill_file)
    )
    skill_name = skill_file.parent.name
    return {
        "skill": skill_name,
        "path": rel_path,
        "verdict": verdict,
        "score": total,
        "blocking_count": sum(1 for item in findings if item.get("level") == "error"),
        "findings": findings,
        "dimensions": dimensions,
        "suggestions": _suggestions(dimensions, findings),
        "command": f"python infra/cannbot-skill-reviewer/scripts/review_skill.py {rel_path}",
    }


def _print_markdown(report: dict[str, object]) -> None:
    _emit("## Skill Review Report")
    _emit()
    _emit(f"- Skill: {report['skill']}")
    _emit(f"- Path: {report['path']}")
    _emit(f"- Verdict: {report['verdict']}")
    _emit(f"- Score: {report['score']}/100")
    _emit(f"- Blocking findings: {report['blocking_count']}")
    _emit()

    _emit("### Validator Findings")
    if report["findings"]:
        _emit("| Level | Rule | Message |")
        _emit("|------|------|---------|")
        for item in report["findings"]:
            msg = str(item.get("msg", "")).replace("|", "\\|")
            _emit(f"| {item.get('level', '')} | {item.get('rule', '')} | {msg} |")
    else:
        _emit("No validator findings.")
    _emit()

    _emit("### Quality Scores")
    _emit("| Dimension | Weight | Score | Reason |")
    _emit("|----------|-------:|------:|--------|")
    for item in report["dimensions"]:
        _emit(f"| {item['name']} | {item['weight']} | {item['score']} | {item['reason']} |")
    _emit()

    _emit("### Required Fixes And Suggestions")
    for idx, tip in enumerate(report["suggestions"], start=1):
        _emit(f"{idx}. {tip}")
    _emit()

    _emit("### Verification")
    _emit(f"- Command: `{report['command']}`")
    _emit(f"- Result: {report['verdict']}")


def main(argv: list[str] | None = None) -> int:
    """Run the skill reviewer CLI."""
    _configure_stream_logger(OUTPUT_LOGGER, sys.stdout)
    _configure_stream_logger(ERROR_LOGGER, sys.stderr)

    parser = argparse.ArgumentParser(description="Review a CANNBot SKILL.md before submission.")
    parser.add_argument("path", help="Path to a skill directory or SKILL.md")
    parser.add_argument("--repo-root", help="Repository root. Auto-detected by default.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    try:
        skill_file = _resolve_skill_file(args.path)
        repo_root = (
            Path(args.repo_root).resolve()
            if args.repo_root
            else _find_repo_root(skill_file)
        )
        report = _build_report(skill_file, repo_root)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        error = {"error": str(exc)}
        if args.json:
            _emit(json.dumps(error, ensure_ascii=False, indent=2))
        else:
            ERROR_LOGGER.error("ERROR: %s", exc)
        return 2

    if args.json:
        _emit(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_markdown(report)

    return 1 if report["verdict"] == "REJECT" else 0


if __name__ == "__main__":
    sys.exit(main())
