# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""PostToolUse hook: 标记 progress.md 已读（配合 pre_tool_use.py 检查 2）"""

import json
import os
import sys


def main():
    data = json.load(sys.stdin)

    agent_id = data.get("agent_id", "")
    if not agent_id:
        sys.exit(0)

    file_path = data.get("tool_input", {}).get("file_path", "")
    if not file_path or not file_path.endswith("progress.md"):
        sys.exit(0)

    marker = f"/tmp/hook_read_progress_{agent_id}.marker"
    with open(marker, "w") as f:
        f.write(file_path)

    sys.exit(0)


if __name__ == "__main__":
    main()
