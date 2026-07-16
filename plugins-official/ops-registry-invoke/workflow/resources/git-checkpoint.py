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
"""Git checkpoint executor for ops-registry-invoke workflow.

This is a standalone CLI script (not an importable module): the workflow
references it directly by path, so the hyphenated filename ``git-checkpoint.py``
is intentional and must be kept — the G.NAM.01 module-name rule does not apply
to executable entry-point scripts invoked via ``python3 <path>``.

Agent passes --operator and --stage; the script looks up a hardcoded
STAGE_TABLE mapping and executes the corresponding git command chain
deterministically.  No git command string assembly by the Agent.

Usage:
    python3 workflow/resources/git-checkpoint.py --operator add --stage 1.1
    python3 workflow/resources/git-checkpoint.py --operator add --stage iter --iteration 2
    python3 workflow/resources/git-checkpoint.py --operator add --stage 4.3 --cwd /path/to/repo
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_RESOURCES_DIR = Path(__file__).resolve().parent
if str(_RESOURCES_DIR) not in sys.path:
    sys.path.insert(0, str(_RESOURCES_DIR))
from _output_log import get_logger

_LOGGER = get_logger("ops_registry_invoke.git_checkpoint")

STAGE_TABLE = {
    "1.1": {
        "commands": [
            ("git", "checkout", "-b", "operators/{op}"),
            ("git", "add", "operators/{op}/"),
            ("git", "commit", "-m", "feat({op}): 1.1 开发准备完成"),
        ],
    },
    "1.2": {
        "commands": [
            ("git", "add", "operators/{op}/"),
            ("git", "commit", "-m", "feat({op}): 需求分析完成"),
            ("git", "tag", "operators/{op}/requirements-approved"),
        ],
    },
    "1.2.5": {
        "commands": [
            ("git", "add", "operators/{op}/"),
            ("git", "commit", "-m", "feat({op}): spec.yaml 生成与 9-stage 校验通过"),
        ],
    },
    "CP1.5": {
        "commands": [
            ("git", "add", "operators/{op}/"),
            ("git", "commit", "-m", "feat({op}): spec.yaml 评审通过 + CP1.5 确认"),
            ("git", "tag", "operators/{op}/spec-approved"),
        ],
    },
    "1.4": {
        "commands": [
            ("git", "add", "operators/{op}/"),
            ("git", "commit", "-m", "feat({op}): 方案设计与测试设计完成"),
            ("git", "tag", "operators/{op}/design-approved"),
        ],
    },
    "W": {
        "commands": [
            ("git", "add", "operators/{op}/"),
            ("git", "commit", "-m", "test({op}): 白盒测试生成与用例汇合完成"),
        ],
    },
    "C": {
        "commands": [
            ("git", "add", "operators/{op}/"),
            ("git", "commit", "-m", "test({op}): PyTorch ST测试开发完成"),
        ],
    },
    "iter": {
        "commands": [
            ("git", "add", "operators/{op}/"),
            ("git", "commit", "-m", "feat({op}): 迭代{i}验收通过"),
            ("git", "tag", "operators/{op}/iter{i}-passed"),
        ],
        "requires_iteration": True,
    },
    "3.1": {
        "commands": [
            ("git", "add", "operators/{op}/"),
            ("git", "commit", "-m", "feat({op}): 精度验收通过"),
            ("git", "tag", "operators/{op}/precision-passed"),
        ],
    },
    "3.2": {
        "commands": [
            ("git", "add", "operators/{op}/"),
            ("git", "commit", "-m", "feat({op}): 性能验收通过"),
            ("git", "tag", "operators/{op}/performance-passed"),
        ],
    },
    "4.3": {
        "commands": [
            ("git", "add", "operators/{op}/"),
            ("git", "commit", "-m", "feat({op}): 上库完成"),
            ("git", "tag", "operators/{op}/done"),
            ("git", "checkout", "main"),
            ("git", "merge", "operators/{op}", "--no-ff", "-m", "feat({op}): 合并算子开发分支"),
            ("git", "checkout", "operators/{op}"),
        ],
    },
}


def run_checkpoint(operator: str, stage: str, iteration: int | None, cwd: str | None) -> int:
    if stage not in STAGE_TABLE:
        _LOGGER.info(json.dumps({
            "success": False,
            "error": f"未知阶段: {stage}，可选: {sorted(STAGE_TABLE.keys())}",
        }))
        return 1

    entry = STAGE_TABLE[stage]

    if entry.get("requires_iteration") and iteration is None:
        _LOGGER.info(json.dumps({
            "success": False,
            "error": f"阶段 {stage} 需要 --iteration 参数",
        }))
        return 1

    i_str = str(iteration) if iteration is not None else ""
    results = []

    for cmd_tuple in entry["commands"]:
        cmd = [s.format(op=operator, i=i_str) for s in cmd_tuple]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        results.append({
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        })
        if proc.returncode != 0:
            _LOGGER.info(json.dumps({
                "success": False,
                "failed_at": cmd,
                "results": results,
            }, ensure_ascii=False))
            return 1

    _LOGGER.info(json.dumps({
        "success": True,
        "operator": operator,
        "stage": stage,
        "results": results,
    }, ensure_ascii=False))
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", required=True, help="算子名（snake_case），如 add, matmul")
    parser.add_argument("--stage", required=True, choices=sorted(STAGE_TABLE.keys()),
                        help="阶段标识，如 1.1, 3.1, iter")
    parser.add_argument("--iteration", type=int, default=None,
                        help="迭代编号 ∈ {1,2,3}，仅 iter 阶段需要")
    parser.add_argument("--cwd", default=None, help="git 命令执行目录（默认当前目录）")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    return run_checkpoint(args.operator, args.stage, args.iteration, args.cwd)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
