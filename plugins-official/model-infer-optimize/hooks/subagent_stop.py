# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""SubagentStop hook: implementer 自验证检查 + 外循环重试限制"""

import json
import logging
import os
import re
import sys

logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

SELF_VERIFY_KEYWORDS_FULL = ["参考 skill", "代码加载", "编译", "推理", "输出"]
SELF_VERIFY_KEYWORDS_DEBUG = ["编译", "推理"]
MAX_RETRY_COUNT = 10  # implementer + reviewer 各 5 轮

SELF_VERIFY_TEMPLATE = """自验证不完整：progress.md 缺少以下自验证项：{missing}。
请完成对应的验证步骤（而非仅补充文字记录），确认结果后写入 progress.md，然后再结束。
格式参考：
### 自验证结果
- 参考 skill: /xxx
- 代码加载: 确认推理加载的是修改后的模型模块和正确的模型配置
- 编译: 通过/失败
- 推理: 通过/失败
- 输出: 合理/异常"""

RETRY_LIMIT_MSG = "重试上限：当前阶段已执行 {n} 轮 implementer/reviewer 循环，超过 5 轮上限。请回退当前阶段改动，向用户报告阻塞点。"


def find_progress_md(cwd):
    """找最近修改的 progress.md（纯 Python 实现，无外部命令依赖）"""
    candidates = []
    try:
        for dirpath, _dirs, filenames in os.walk(cwd):
            if "/.git/" in dirpath or dirpath.endswith("/.git"):
                continue
            if "progress.md" in filenames:
                candidates.append(os.path.join(dirpath, "progress.md"))
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return candidates[0]


def get_current_stage(content):
    """从 progress.md 中提取当前阶段号"""
    matches = re.findall(r"## 阶段\s*(\d+)", content)
    return matches[-1] if matches else "0"


def check_self_verification(data, progress_path, content):
    """检查 1：Implementer 自验证检查（P0，阻断）"""
    if data.get("agent_type") != "model-infer-implementer":
        return None

    is_debug = "### 调试记录" in content and "[修复]" in content
    keywords = SELF_VERIFY_KEYWORDS_DEBUG if is_debug else SELF_VERIFY_KEYWORDS_FULL

    if "### 自验证结果" not in content:
        missing = "、".join(keywords)
        return SELF_VERIFY_TEMPLATE.format(missing=missing)

    verify_start = content.index("### 自验证结果")
    verify_section = content[verify_start:]
    next_section = verify_section.find("\n### ", 1)
    if next_section > 0:
        verify_section = verify_section[:next_section]

    missing = [kw for kw in keywords if kw not in verify_section]
    if missing:
        return SELF_VERIFY_TEMPLATE.format(missing="、".join(missing))

    return None


def check_retry_limit(data, content):
    """检查 2：外循环重试限制（P1，阻断）"""
    agent_type = data.get("agent_type", "")
    if agent_type not in ("model-infer-implementer", "model-infer-reviewer"):
        return None

    session_id = data.get("session_id", "unknown")
    stage = get_current_stage(content)
    counter_file = f"/tmp/hook_retry_{session_id}_{stage}.count"

    count = 0
    if os.path.exists(counter_file):
        try:
            with open(counter_file) as f:
                count = int(f.read().strip())
        except (ValueError, OSError):
            count = 0

    count += 1
    with open(counter_file, "w") as f:
        f.write(str(count))

    if count > MAX_RETRY_COUNT:
        rounds = count // 2
        return RETRY_LIMIT_MSG.format(n=rounds)

    return None


def main():
    data = json.load(sys.stdin)
    cwd = data.get("cwd", ".")

    progress_path = find_progress_md(cwd)
    if not progress_path:
        sys.exit(0)

    try:
        with open(progress_path) as f:
            content = f.read()
    except OSError:
        sys.exit(0)

    for check in [
        lambda: check_self_verification(data, progress_path, content),
        lambda: check_retry_limit(data, content),
    ]:
        reason = check()
        if reason:
            logger.error(reason)
            sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
