# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""PreToolUse hook: 长时间任务周期提醒（非阻断注入）"""

import json
import os
import sys
import time

REMIND_AFTER_MINUTES = 60
REMIND_INTERVAL_MINUTES = 30


def main():
    data = json.load(sys.stdin)

    agent_id = data.get("agent_id", "")
    if not agent_id:
        sys.exit(0)

    now = time.time()
    start_file = f"/tmp/hook_start_{agent_id}.ts"
    remind_file = f"/tmp/hook_remind_{agent_id}.ts"

    if not os.path.exists(start_file):
        with open(start_file, "w") as f:
            f.write(str(now))
        sys.exit(0)

    with open(start_file) as f:
        start_time = float(f.read().strip())

    elapsed_minutes = (now - start_time) / 60

    if elapsed_minutes < REMIND_AFTER_MINUTES:
        sys.exit(0)

    if os.path.exists(remind_file):
        with open(remind_file) as f:
            last_remind = float(f.read().strip())
        if (now - last_remind) / 60 < REMIND_INTERVAL_MINUTES:
            sys.exit(0)

    with open(remind_file, "w") as f:
        f.write(str(now))

    elapsed = int(elapsed_minutes)
    result = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "additionalContext": (
                f"你已运行超过 {elapsed} 分钟。请确认："
                "1) 是否按 skill 流程的实施/调试步骤要求执行 "
                "2) 关键进展是否已写入 progress.md "
                "3) 审视当前改动方向是否正确：实施偏差应自行纠正，方案本身不可行应停止并返回主流程报告阻塞。"
            ),
        }
    }
    json.dump(result, sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
