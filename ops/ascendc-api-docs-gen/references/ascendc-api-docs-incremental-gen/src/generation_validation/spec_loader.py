# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""写作规范的来源：直接复用 asc-devkit 仓 docs/zh 下的规范文件。

本仓不再保留规范副本（原 asc_writing_spec.md 已删除）。生成文档时按章节标题
从 devkit 规范里取对应小节，注入到写作提示词中。

规范目录解析顺序：
1. 环境变量 ``ASC_DEVKIT_DOCS_DIR``（指向 devkit 的 docs/zh 目录）；
2. ``<repo>/data/asc-devkit-master/docs/zh``、``<repo>/data/asc-devkit/docs/zh``；
3. ``<repo>/.temp_git/*/now/docs/zh``（run_by_pr_test 检出的 PR worktree）。
取不到时只告警一次，返回空串，由调用方退回本地补充规则。
"""

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

DOCS_DIR_ENV = "ASC_DEVKIT_DOCS_DIR"

# 规范文件（kind -> 文件名）
SPEC_FILES = {
    "writing": "asc_api_writing_guidelines.md",
    "design": "asc_doc_design_rules.md",
}

_logger = logging.getLogger(__name__)
_warned: set[str] = set()


def _repo_root() -> Path:
    from .config import PROJECT_ROOT

    return PROJECT_ROOT


@lru_cache(maxsize=1)
def docs_dir() -> Optional[Path]:
    """定位 devkit 的 docs/zh 目录；找不到返回 None。"""
    env = os.environ.get(DOCS_DIR_ENV)
    if env:
        p = Path(env)
        if p.is_dir():
            return p
        _warn(f"{DOCS_DIR_ENV}={env} 不存在或不是目录")

    root = _repo_root()
    for cand in (
        root / "data" / "asc-devkit-master" / "docs" / "zh",
        root / "data" / "asc-devkit" / "docs" / "zh",
    ):
        if cand.is_dir():
            return cand

    for cand in sorted(root.glob(".temp_git/*/now/docs/zh")):
        if cand.is_dir():
            return cand

    _warn("未找到 asc-devkit 的 docs/zh 目录（可设 ASC_DEVKIT_DOCS_DIR）")
    return None


def _warn(msg: str) -> None:
    if msg in _warned:
        return
    _warned.add(msg)
    _logger.warning("[spec_loader] %s", msg)


@lru_cache(maxsize=4)
def spec_text(kind: str = "writing") -> str:
    """读取规范文件全文；不可用时返回空串。"""
    name = SPEC_FILES.get(kind)
    if not name:
        raise ValueError(f"未知规范类型: {kind}")
    d = docs_dir()
    if d is None:
        return ""
    path = d / name
    if not path.is_file():
        _warn(f"规范文件不存在: {path}")
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


_SECTION_TAIL = re.compile(r"<a\s+name=[^>]*></a>", re.IGNORECASE)


@lru_cache(maxsize=32)
def section(heading: str, kind: str = "writing") -> str:
    """按二级标题（``## <heading>``）抽取章节正文，去掉标题里的锚点。

    ``heading`` 支持前缀匹配（例如 "6 参数说明规范"）。
    """
    text = spec_text(kind)
    if not text:
        return ""
    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if not line.startswith("## "):
            continue
        title = _SECTION_TAIL.sub("", line[3:]).strip()
        if title == heading or title.startswith(heading):
            start = i
            break
    if start is None:
        _warn(f"规范中未找到章节: ## {heading}")
        return ""

    body = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        body.append(line)
    return "\n".join(body).strip()


def available() -> bool:
    """规范是否可用（供测试/日志判断）。"""
    return bool(spec_text("writing"))


def clear_cache() -> None:
    """清空缓存（测试或切换规范目录时使用）。"""
    docs_dir.cache_clear()
    spec_text.cache_clear()
    section.cache_clear()
    _warned.clear()
