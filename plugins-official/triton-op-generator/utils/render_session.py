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

"""将 Claude Code session jsonl 渲染为可读 markdown。

用法:
    python render_session.py <session.jsonl> <session.md>
"""
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def _extract_text_block(block):
    """提取单个文本块的内容。

    Args:
        block: 消息块字典

    Returns:
        str: 提取的文本内容
    """
    if not isinstance(block, dict):
        return ""

    btype = block.get("type")
    if btype == "text":
        return block.get("text", "")

    if btype == "tool_use":
        name = block.get("name", "?")
        inp = block.get("input", {})
        try:
            inp_s = json.dumps(inp, ensure_ascii=False)[:500]
        except Exception:
            inp_s = str(inp)[:500]
        return f"[tool_use: {name}] {inp_s}"

    if btype == "tool_result":
        out = block.get("content", "")
        if isinstance(out, list):
            out = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in out
            )
        return f"[tool_result]\n{str(out)[:2000]}"

    return ""


def _extract_text(message):
    """从消息中提取文本内容。

    Args:
        message: 消息对象（字符串或字典）

    Returns:
        str: 提取的文本内容
    """
    if isinstance(message, str):
        return message

    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = [_extract_text_block(block) for block in content]
        return "\n\n".join(p for p in parts if p)

    return ""


def _process_message_entry(msg_type, message_data):
    """处理单条消息条目。

    Args:
        msg_type: 消息类型（user/assistant）
        message_data: 消息数据字典

    Returns:
        str: 格式化的消息文本，无内容时返回空字符串
    """
    text = _extract_text(message_data)
    if not text.strip():
        return ""

    if msg_type == "user":
        return f"## * User\n\n{text}\n"
    if msg_type == "assistant":
        return f"## * Assistant\n\n{text}\n"
    return ""


def render(src: Path, dst: Path):
    """渲染 Claude Code session 文件。

    Args:
        src: 源 jsonl 文件路径
        dst: 目标 markdown 文件路径
    """
    lines = ["# Claude Code Session\n"]

    with src.open(encoding="utf-8") as f:
        for raw in f:
            try:
                d = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning(f"跳过无效的 JSON 行: {e}")
                continue

            msg_type = d.get("type")
            formatted = _process_message_entry(msg_type, d.get("message", {}))
            if formatted:
                lines.append(formatted)

    dst.write_text("\n".join(lines), encoding="utf-8")


def main():
    """主函数。"""
    if len(sys.argv) != 3:
        logger.error("Usage: render_session.py <src.jsonl> <dst.md>")
        sys.exit(2)

    src_path = Path(sys.argv[1])
    dst_path = Path(sys.argv[2])

    if not src_path.exists():
        logger.error(f"源文件不存在: {src_path}")
        sys.exit(1)

    render(src_path, dst_path)
    logger.info(f"Rendered: {dst_path}")


if __name__ == "__main__":
    main()
