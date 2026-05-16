# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""归档 progress.md 的工作区内容到 progress_history.md。

常驻区与工作区用以下标记分隔：
    <!-- ===== 以上为常驻区，不清除 ===== -->
    <!-- ===== 以下为工作区，阶段推进时归档清空 ===== -->
"""

import logging
import os
import sys
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SEPARATOR_END = "<!-- ===== 以上为常驻区，不清除 ===== -->"
SEPARATOR_START = "<!-- ===== 以下为工作区，阶段推进时归档清空 ===== -->"


def archive_progress(progress_path: str) -> None:
    if not os.path.exists(progress_path):
        logger.error("文件不存在: %s", progress_path)
        sys.exit(1)

    with open(progress_path, "r", encoding="utf-8") as f:
        content = f.read()

    if SEPARATOR_END not in content:
        logger.error("未找到常驻区结束标记，跳过归档。")
        logger.error("请确保 progress.md 包含标记：%s", SEPARATOR_END)
        sys.exit(1)

    parts = content.split(SEPARATOR_END, 1)
    persistent_section = parts[0] + SEPARATOR_END

    work_section = parts[1] if len(parts) > 1 else ""
    if SEPARATOR_START in work_section:
        work_section = work_section.split(SEPARATOR_START, 1)[1]
    work_section = work_section.strip()

    if not work_section:
        logger.info("工作区为空，无需归档。")
        return

    # 归档到 progress_history.md
    progress_dir = os.path.dirname(os.path.abspath(progress_path))
    history_path = os.path.join(progress_dir, "progress_history.md")
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    archive_entry = f"\n\n---\n\n## 归档于 {timestamp}\n\n{work_section}\n"

    if not os.path.exists(history_path):
        persistent_text = parts[0].strip()
        header = (
            "<!-- 本文件默认禁止全文读取。需要历史信息时请用 Grep 按关键字查找。 -->\n"
            f"# 进度历史归档\n\n## 常驻区快照\n\n{persistent_text}\n"
        )
        with open(history_path, "w", encoding="utf-8") as f:
            f.write(header)

    with open(history_path, "a", encoding="utf-8") as f:
        f.write(archive_entry)

    # 清空工作区
    new_content = f"{persistent_section}\n\n{SEPARATOR_START}\n\n"
    with open(progress_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    work_lines = len(work_section.splitlines())
    logger.info("归档完成：%d 行 → %s", work_lines, history_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python3 archive_progress.py <progress.md>")
        sys.exit(1)
    archive_progress(sys.argv[1])
