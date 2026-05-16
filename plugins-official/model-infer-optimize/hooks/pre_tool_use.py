# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""PreToolUse hook: 角色越界文件保护 + 改代码前必读 progress.md"""

import fnmatch
import json
import logging
import os
import sys

logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

# 保护模式：agent_type -> 不允许修改的文件模式列表
PROTECTED_PATTERNS = {
    "model-infer-analyzer": [
        "modeling_*.py",
        "runner_*.py",
        "*.yaml",
        "executor/*",
    ],
    "model-infer-reviewer": [
        "modeling_*.py",
        "runner_*.py",
    ],
}

# 改代码前必读 progress.md 的文件模式
CODE_PATTERNS = ["modeling_*.py", "runner_*.py"]


def matches_any(basename, full_path, patterns):
    for pat in patterns:
        if fnmatch.fnmatch(basename, pat):
            return True
        if "/" in pat and fnmatch.fnmatch(full_path, f"*/{pat}"):
            return True
    return False


def check_role_protection(data):
    """检查 1：角色越界文件保护（P0，阻断）"""
    agent_type = data.get("agent_type", "")
    agent_id = data.get("agent_id", "")
    file_path = data.get("tool_input", {}).get("file_path", "")

    if not agent_id or not file_path:
        return None

    patterns = PROTECTED_PATTERNS.get(agent_type)
    if not patterns:
        return None

    basename = os.path.basename(file_path)
    if matches_any(basename, file_path, patterns):
        return f"禁止：{agent_type} 不允许修改 {basename}。模型代码修改请通过 implementer 执行。"

    return None


def check_read_progress_first(data):
    """检查 2：改代码前必读 progress.md（P1，阻断）"""
    agent_id = data.get("agent_id", "")
    file_path = data.get("tool_input", {}).get("file_path", "")

    if not agent_id or not file_path:
        return None

    basename = os.path.basename(file_path)
    if not matches_any(basename, file_path, CODE_PATTERNS):
        return None

    marker = f"/tmp/hook_read_progress_{agent_id}.marker"
    if os.path.exists(marker):
        return None

    return "禁止：修改模型代码前必须先读取 progress.md，了解当前阶段方案和实施记录后再修改代码。"


def main():
    data = json.load(sys.stdin)

    for check in [check_role_protection, check_read_progress_first]:
        reason = check(data)
        if reason:
            logger.error(reason)
            sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
