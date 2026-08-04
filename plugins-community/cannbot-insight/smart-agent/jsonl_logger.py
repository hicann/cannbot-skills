# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Claude Code JSONL 格式日志器。

逐行写 JSONL，字段与 Claude Code session 日志（~/.claude/projects/*/*.jsonl）一致，
CANNBot-Insight 的 claude-jsonl 适配器可直接解析。

行类型映射：
  - 初始 prompt          → user 消息（content=string）
  - LLM 响应（JSON）      → assistant 消息（content=[{type:text}]）
  - LLM 响应（read 请求）  → assistant 消息（content=[{type:tool_use,name:read}]）
  - read 兑现             → user 消息（content=[{type:tool_result}]）
  - schema 校验失败重试    → user 消息（content=string）
  - 进度/心跳             → system 行（subtype=progress，适配器跳过但留档）
  - 最终结果             → result 行
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION = "smart-agent-1.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _git_branch(cwd: str) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=3,
        )
    except Exception:
        return ""
    if r.returncode == 0:
        return r.stdout.strip()
    return ""


class JsonlLogger:
    """逐行写 Claude Code 格式 JSONL。"""

    def __init__(self, log_path: str, session_id: str | None = None, cwd: str | None = None):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or _new_uuid()
        self.cwd = cwd or os.getcwd()
        self.git_branch = _git_branch(self.cwd)
        self._parent_uuid: str | None = None
        self._fh = open(self.log_path, "a", encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        if self._fh and not self._fh.closed:
            self._fh.close()

    # ---- 消息行 ----

    def log_user_text(self, content: str, **extra: Any) -> str:
        """用户文本消息（初始 prompt / schema 重试指令）。"""
        return self._write({
            "type": "user",
            "message": {"role": "user", "content": content},
            "cwd": self.cwd,
            "gitBranch": self.git_branch,
            "userType": "external",
            "entrypoint": "api",
            **extra,
        })

    def log_assistant(self, content_blocks: list[dict], *, model: str = "",
                      usage: dict | None = None, duration_ms: int = 0,
                      stop_reason: str = "end_turn", **extra: Any) -> str:
        """assistant 消息。content_blocks 为 [{type:text|tool_use|thinking, ...}]。"""
        msg: dict = {
            "id": _new_uuid(),
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": None,
        }
        if model:
            msg["model"] = model
        if usage:
            msg["usage"] = usage
        obj = {"type": "assistant", "message": msg}
        if duration_ms:
            obj["duration_ms"] = duration_ms
        obj["cwd"] = self.cwd
        obj["gitBranch"] = self.git_branch
        obj["userType"] = "external"
        obj["entrypoint"] = "api"
        return self._write({**obj, **extra})

    def log_tool_result(self, tool_use_id: str, content: str, *,
                        is_error: bool = False, source_uuid: str | None = None,
                        **extra: Any) -> str:
        """user 行携带 tool_result（read 兑现的 snippet）。"""
        block: dict = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            block["is_error"] = True
        obj = {
            "type": "user",
            "message": {"role": "user", "content": [block]},
            "cwd": self.cwd,
            "gitBranch": self.git_branch,
            "userType": "external",
            "entrypoint": "api",
        }
        if source_uuid:
            obj["sourceToolAssistantUUID"] = source_uuid
        return self._write({**obj, **extra})

    # ---- 元数据行 ----

    def log_system(self, subtype: str, data: dict | None = None, **extra: Any) -> str:
        """system 行（进度/心跳，适配器跳过但留档）。"""
        obj: dict = {"type": "system", "subtype": subtype, "isMeta": False}
        if data:
            obj["data"] = data
        obj["cwd"] = self.cwd
        obj["gitBranch"] = self.git_branch
        obj["userType"] = "external"
        obj["entrypoint"] = "api"
        return self._write({**obj, **extra})

    def log_result(self, result: str, *, subtype: str = "success",
                   cost_usd: float = 0, duration_ms: int = 0, **extra: Any) -> str:
        obj = {"type": "result", "subtype": subtype, "result": result}
        if cost_usd:
            obj["cost_usd"] = cost_usd
        if duration_ms:
            obj["duration_ms"] = duration_ms
        return self._write({**obj, **extra})

    def _write(self, obj: dict) -> str:
        obj.setdefault("uuid", _new_uuid())
        obj.setdefault("parentUuid", self._parent_uuid)
        obj.setdefault("isSidechain", False)
        obj.setdefault("sessionId", self.session_id)
        obj.setdefault("timestamp", _now_iso())
        obj.setdefault("version", VERSION)
        uid = obj["uuid"]
        self._parent_uuid = uid
        line = json.dumps(obj, ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()
        return uid

