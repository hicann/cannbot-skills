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

"""PostToolUse hook: when errors occur during dsl-lowering / ascendc-evaluation,
suggest relevant debug skills to Claude Code.

NOTE: hooks.json `matcher` only supports tool-name filtering (e.g. "Bash").
Command-level filtering is done here via a cheap string pre-check before any
regex or JSON parsing, so unrelated Bash calls exit in < 1ms.
"""
import sys
import json
import re
import logging
import argparse
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fast pre-filter: only read full JSON if the raw stdin *might* be relevant.
# These are plain substrings checked with `in`, no regex overhead.
# ---------------------------------------------------------------------------
FAST_CMD_KEYWORDS = (
    # dsl-lowering — compilation
    "build.sh",
    "cmake",
    "make",
    # dsl-lowering pass names
    "tiling_pass",
    "init_pass",
    "process_pass",
    "process_nonaligned_pass",
    # ascendc-evaluation — install .run & run evaluate
    ".run",
    "evaluate.py",
    "precision_results",
    "evaluation_result",
)

# ---------------------------------------------------------------------------
# Stage detection (regex, only runs after pre-filter passes)
# ---------------------------------------------------------------------------
STAGE_PATTERNS = {
    "dsl-lowering": [
        re.compile(r"dsl.lowering", re.IGNORECASE),
        re.compile(r"tiling_pass|init_pass|process_pass|process_nonaligned_pass", re.IGNORECASE),
        re.compile(r"build\.sh", re.IGNORECASE),
        re.compile(r"cmake|make\b", re.IGNORECASE),
    ],
    "ascendc-evaluation": [
        re.compile(r"evaluate\.py", re.IGNORECASE),
        re.compile(r"ascendc.evaluation", re.IGNORECASE),
        re.compile(r"precision_results|evaluation_result", re.IGNORECASE),
        re.compile(r"\.run\b"),  # installing .run packages
    ],
}

# ---------------------------------------------------------------------------
# Error detection
# ---------------------------------------------------------------------------
ERROR_PATTERNS = [
    re.compile(r"\berror\b", re.IGNORECASE),
    re.compile(r"\bfailed\b", re.IGNORECASE),
    re.compile(r"Exit code [1-9]"),
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"FAILED|ABORT|Segmentation fault|core dumped"),
    re.compile(r"precision.*fail|accuracy.*fail|mismatch", re.IGNORECASE),
    re.compile(r"RuntimeError|TypeError|ValueError|KeyError|IndexError"),
    re.compile(r"compile.*error|link.*error|undefined reference", re.IGNORECASE),
    re.compile(r"speedup.*below|perf.*regression|slower than", re.IGNORECASE),
    re.compile(r"No such file or directory", re.IGNORECASE),
    re.compile(r"Permission denied", re.IGNORECASE),
    re.compile(r"install.*fail|setup.*fail", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Skill recommendations per (stage, error_type)
# ---------------------------------------------------------------------------
SKILL_RECOMMENDATIONS = {
    "dsl-lowering": {
        "compile_error": [
            "cake-docs-search — look up API docs and usage examples",
            "cake-code-review — check for coding red-line violations that cause build failures",
        ],
        "runtime_error": [
            "ascendc-op-debug — diagnose runtime crashes, precision errors, hangs with hypothesis-driven protocol",
            "cake-docs-search — look up API constraints and correct usage",
        ],
    },
    "ascendc-evaluation": {
        "install_error": [
            "remote-cann-development — check CANN environment and .run package installation",
            "cake-docs-search — verify installation prerequisites and environment setup",
        ],
        "precision_error": [
            "ascendc-op-debug — diagnose precision errors with 14 hypothesis patterns (cross-tile accumulation, einsum semantics, etc.)",
            "op-dashboard — generate an HTML dashboard to inspect precision results",
            "code-performance-advisor — check if precision issue is related to optimization tradeoffs",
        ],
        "performance_error": [
            "code-performance-advisor — diagnose performance bottlenecks from profiling data",
            "op-dashboard — generate an HTML dashboard to inspect performance results",
        ],
        "runtime_error": [
            "ascendc-op-debug — diagnose runtime crashes/hangs during evaluation",
            "remote-cann-development — check CANN environment if environment-related failure",
        ],
    },
}


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def detect_stage(cmd: str, resp: str) -> Optional[str]:
    combined = cmd + " " + resp
    for stage, patterns in STAGE_PATTERNS.items():
        for pat in patterns:
            if pat.search(combined):
                return stage
    return None


def detect_error_type(cmd: str, resp: str, exit_code: int = 0) -> Optional[str]:
    has_error_text = any(pat.search(resp) for pat in ERROR_PATTERNS)
    if not has_error_text and exit_code == 0:
        return None

    # Install errors (.run packages, pip install, setup.sh)
    if re.search(r"\.run\b", cmd) or re.search(
            r"install.*fail|setup.*fail|Permission denied|No such file", resp, re.IGNORECASE):
        return "install_error"
    if re.search(r"precision.*fail|accuracy.*fail|mismatch|cosine.*similarity|relative.*error", resp, re.IGNORECASE):
        return "precision_error"
    if re.search(r"compile.*error|link.*error|undefined reference|syntax error|undeclared|no member", resp, re.IGNORECASE):
        return "compile_error"
    if re.search(r"speedup.*below|perf.*regression|slower than", resp, re.IGNORECASE):
        return "performance_error"
    return "runtime_error"


def build_advice(stage: str, error_type: str) -> str:
    skills = SKILL_RECOMMENDATIONS.get(stage, {})
    recommendations = skills.get(error_type, skills.get("runtime_error", []))

    lines = [
        f"[debug-skill-advisor] Detected {error_type.replace('_', ' ')} during {stage}.",
        "Recommended debug skills (invoke via Skill tool):",
    ]
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"  {i}. {rec}")
    lines.append("Tip: Start with the first recommendation.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Hook diagnostics must stay on stderr so protocol stdout (JSON) is
    # byte-for-byte unchanged for the upstream parser.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", default="PostToolUse",
                        choices=["PostToolUse", "PostToolUseFailure"],
                        help="Hook event name passed from hooks.json")
    args, _ = parser.parse_known_args()
    hook_event = args.event

    try:
        raw = sys.stdin.read()

        # ---- fast pre-filter: skip if no relevant keyword in raw input ----
        if not any(kw in raw for kw in FAST_CMD_KEYWORDS):
            return

        data = json.loads(raw)
        cmd = data.get("tool_input", {}).get("command", "")

        # tool_response is an object; Claude Code sends {"stdout": "...", "stderr": "..."}
        # but documented format uses {"output": "...", "exit_code": N} — handle both
        tool_response = data.get("tool_response", {})
        if isinstance(tool_response, dict):
            resp = tool_response.get("output") or tool_response.get("stdout", "")
            exit_code = int(tool_response.get("exit_code", 0) or 0)
        else:
            resp = str(tool_response)
            exit_code = 0

        stage = detect_stage(cmd, resp)
        if not stage:
            return

        error_type = detect_error_type(cmd, resp, exit_code)
        if not error_type:
            return

        advice = build_advice(stage, error_type)
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": hook_event,
                "additionalContext": advice,
            },
            "systemMessage": advice,
        }))

    except Exception as e:
        logger.error(f"[debug-skill-advisor] error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
