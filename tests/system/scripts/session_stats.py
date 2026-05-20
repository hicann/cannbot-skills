# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
OpenCode Session Stats - 统计 OpenCode Session 的 token 消耗、工具调用等

支持两种数据源:
1. 从导出的 JSON 文件加载
2. 直接从 OpenCode SQLite 数据库查询

使用示例:
    # 从导出文件加载
    stats = SessionStats.from_export_file("session_export.json")

    # 从数据库直接查询
    stats = SessionStats.from_database("ses_xxx")

    # 打印统计报告
    stats.print_report()
"""

import json
import logging
import sqlite3
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


@dataclass
class TokenUsage:
    total: int = 0
    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def add(self, other: Dict[str, int]):
        self.total += other.get("total", 0)
        self.input += other.get("input", 0)
        self.output += other.get("output", 0)
        self.reasoning += other.get("reasoning", 0)
        cache = other.get("cache", {})
        self.cache_read += cache.get("read", 0)
        self.cache_write += cache.get("write", 0)


@dataclass
class SessionStats:
    session_id: str
    title: str = ""
    directory: str = ""

    tokens: TokenUsage = field(default_factory=TokenUsage)
    cost: float = 0.0

    tool_calls: Dict[str, int] = field(default_factory=dict)
    total_steps: int = 0
    stop_reason: str = ""

    test_runs: int = 0
    test_commands: List[str] = field(default_factory=list)

    files_modified: int = 0
    additions: int = 0
    deletions: int = 0

    time_created: int = 0
    time_updated: int = 0

    duration_ms: int = 0
    reasoning_time_ms: int = 0
    tool_time_ms: int = 0
    output_time_ms: int = 0

    # ---- Public class-level API ----

    @classmethod
    def from_export_file(cls, file_path: str) -> "SessionStats":
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        return cls._parse_export_data(data)

    @classmethod
    def from_database(cls, session_id: str, db_path: Optional[str] = None) -> "SessionStats":
        if db_path is None:
            db_path = cls._get_default_db_path()

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        stats = cls._query_database(conn, session_id)
        conn.close()

        return stats

    # ---- Public instance methods ----

    def print_report(self):
        logger.info("\n%s", "=" * 60)
        logger.info("Session Stats Report: %s", self.session_id)
        logger.info("=" * 60)

        logger.info("\n[基本信息]")
        logger.info("  标题: %s", self.title)
        logger.info("  目录: %s", self.directory)
        logger.info("  创建时间: %s", self._format_time(self.time_created))
        logger.info("  更新时间: %s", self._format_time(self.time_updated))

        logger.info("\n[耗时统计]")
        logger.info("  总耗时: %s", self._format_duration(self.duration_ms))
        logger.info("  Reasoning耗时: %s", self._format_duration(self.reasoning_time_ms))
        logger.info("  工具执行耗时: %s", self._format_duration(self.tool_time_ms))
        logger.info("  输出生成耗时: %s", self._format_duration(self.output_time_ms))

        logger.info("\n[Token 统计]")
        logger.info("  Total: %s", self.tokens.total)
        logger.info("  Input: %s", self.tokens.input)
        logger.info("  Output: %s", self.tokens.output)
        logger.info("  Reasoning: %s", self.tokens.reasoning)
        logger.info("  Cache Read: %s", self.tokens.cache_read)
        logger.info("  Cache Write: %s", self.tokens.cache_write)
        logger.info("  总成本: $%.6f", self.cost)

        logger.info("\n[执行统计]")
        logger.info("  总迭代步数: %s", self.total_steps)
        logger.info("  结束原因: %s", self.stop_reason)

        logger.info("\n[工具调用统计]")
        if self.tool_calls:
            for tool, count in sorted(self.tool_calls.items(), key=lambda x: -x[1]):
                logger.info("  %s: %d 次", tool, count)
        else:
            logger.info("  无工具调用")

        logger.info("\n[测试统计]")
        logger.info("  测试运行次数: %s", self.test_runs)
        if self.test_commands:
            for i, cmd in enumerate(self.test_commands, 1):
                logger.info("    #%d: %s...", i, cmd[:80])

        logger.info("\n[代码变更统计]")
        logger.info("  修改文件数: %s", self.files_modified)
        logger.info("  新增行数: %s", self.additions)
        logger.info("  删除行数: %s", self.deletions)

        logger.info("\n%s\n", "=" * 60)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "directory": self.directory,
            "tokens": {
                "total": self.tokens.total,
                "input": self.tokens.input,
                "output": self.tokens.output,
                "reasoning": self.tokens.reasoning,
                "cache_read": self.tokens.cache_read,
                "cache_write": self.tokens.cache_write,
            },
            "cost": self.cost,
            "tool_calls": self.tool_calls,
            "total_steps": self.total_steps,
            "stop_reason": self.stop_reason,
            "test_runs": self.test_runs,
            "test_commands": self.test_commands,
            "files_modified": self.files_modified,
            "additions": self.additions,
            "deletions": self.deletions,
            "time_created": self.time_created,
            "time_updated": self.time_updated,
            "duration_ms": self.duration_ms,
            "duration": self._format_duration(self.duration_ms),
            "reasoning_time_ms": self.reasoning_time_ms,
            "reasoning_time": self._format_duration(self.reasoning_time_ms),
            "tool_time_ms": self.tool_time_ms,
            "tool_time": self._format_duration(self.tool_time_ms),
            "output_time_ms": self.output_time_ms,
            "output_time": self._format_duration(self.output_time_ms),
        }

    def check_test_command(self, command: str):
        """检查命令是否为测试命令"""
        test_keywords = ["pytest", "test", "unittest", "npm test", "go test", "cargo test"]
        for kw in test_keywords:
            if kw in command.lower():
                self.test_runs += 1
                self.test_commands.append(command)
                break

    # ---- Private static helpers ----

    @staticmethod
    def _format_time(timestamp_ms: int) -> str:
        from datetime import datetime
        if timestamp_ms:
            return datetime.fromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
        return "N/A"

    @staticmethod
    def _format_duration(duration_ms: int) -> str:
        if not duration_ms:
            return "N/A"

        seconds = duration_ms / 1000

        if seconds < 60:
            return f"{seconds:.1f}s"

        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60

        if minutes < 60:
            return f"{minutes}m {remaining_seconds:.0f}s"

        hours = minutes // 60
        remaining_minutes = minutes % 60

        return f"{hours}h {remaining_minutes}m {remaining_seconds:.0f}s"

    @staticmethod
    def _get_default_db_path() -> str:
        default_path = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
        if default_path.exists():
            return str(default_path)
        raise FileNotFoundError(f"OpenCode database not found at {default_path}")

    @staticmethod
    def _calc_duration(part_time: Dict) -> int:
        """计算时间差"""
        if not part_time:
            return 0
        start = part_time.get("start", 0)
        end = part_time.get("end", 0)
        return (end - start) if start and end else 0

    # ---- Private class-level helpers ----

    @classmethod
    def _parse_export_data(cls, data: Dict[str, Any]) -> "SessionStats":
        info = data.get("info", {})
        stats = cls(
            session_id=info.get("id", ""),
            title=info.get("title", ""),
            directory=info.get("directory", ""),
            time_created=info.get("time", {}).get("created", 0),
            time_updated=info.get("time", {}).get("updated", 0),
        )

        summary = info.get("summary", {})
        stats.files_modified = summary.get("files", 0)
        stats.additions = summary.get("additions", 0)
        stats.deletions = summary.get("deletions", 0)

        if stats.time_created and stats.time_updated:
            stats.duration_ms = stats.time_updated - stats.time_created

        messages = data.get("messages", [])
        for msg in messages:
            cls._process_message(msg, stats)

        return stats

    @classmethod
    def _process_message(cls, msg: Dict[str, Any], stats: "SessionStats") -> None:
        """处理单条消息，更新统计信息"""
        msg_info = msg.get("info", {})

        if msg_info.get("role") == "assistant":
            cls._process_assistant_info(msg_info, stats)

        parts = msg.get("parts", [])
        for part in parts:
            cls._process_part(part, stats)

    @classmethod
    def _process_assistant_info(cls, msg_info: Dict[str, Any], stats: "SessionStats") -> None:
        """处理 assistant 消息的元信息"""
        tokens = msg_info.get("tokens", {})
        if tokens:
            stats.tokens.add(tokens)

        stats.cost += msg_info.get("cost", 0)
        stats.total_steps += 1

        if msg_info.get("finish") == "stop":
            stats.stop_reason = "stop"

    @classmethod
    def _process_part(cls, part: Dict[str, Any], stats: "SessionStats") -> None:
        """处理消息中的单个 part"""
        part_type = part.get("type")
        part_time = part.get("time", {})

        if part_type == "tool":
            cls._process_tool_part(part, part_time, stats)
        elif part_type == "reasoning":
            stats.reasoning_time_ms += cls._calc_duration(part_time)
        elif part_type == "text":
            stats.output_time_ms += cls._calc_duration(part_time)

    @classmethod
    def _process_tool_part(cls, part: Dict[str, Any], part_time: Dict, stats: "SessionStats") -> None:
        """处理 tool 类型的 part"""
        tool_name = part.get("tool", "unknown")
        stats.tool_calls[tool_name] = stats.tool_calls.get(tool_name, 0) + 1
        stats.tool_time_ms += cls._calc_duration(part_time)

        if tool_name == "bash":
            state = part.get("state", {})
            if isinstance(state.get("input"), dict):
                command = state.get("input", {}).get("command", "")
                stats.check_test_command(command)

    @classmethod
    def _query_session_info(cls, conn: sqlite3.Connection, session_id: str) -> "SessionStats":
        """Query session row and build base stats object."""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, directory, time_created, time_updated, "
            "summary_additions, summary_deletions, summary_files "
            "FROM session WHERE id = ?",
            (session_id,)
        )
        row = cursor.fetchone()

        if not row:
            raise ValueError(f"Session not found: {session_id}")

        stats = cls(
            session_id=row["id"],
            title=row["title"] or "",
            directory=row["directory"] or "",
            time_created=row["time_created"],
            time_updated=row["time_updated"],
            additions=row["summary_additions"] or 0,
            deletions=row["summary_deletions"] or 0,
            files_modified=row["summary_files"] or 0,
        )

        if stats.time_created and stats.time_updated:
            stats.duration_ms = stats.time_updated - stats.time_created

        return stats

    @classmethod
    def _query_step_finish(cls, conn: sqlite3.Connection, session_id: str,
                           stats: "SessionStats") -> None:
        """Process step-finish parts for tokens, cost, and stop reason."""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data FROM part "
            "WHERE session_id = ? AND json_extract(data, '$.type') = 'step-finish'",
            (session_id,)
        )
        for row in cursor.fetchall():
            part_data = json.loads(row["data"])
            stats.total_steps += 1
            if part_data.get("reason") == "stop":
                stats.stop_reason = "stop"
            tokens = part_data.get("tokens", {})
            if tokens:
                stats.tokens.add(tokens)
            stats.cost += part_data.get("cost", 0)

    @classmethod
    def _query_tool_parts(cls, conn: sqlite3.Connection, session_id: str,
                          stats: "SessionStats") -> None:
        """Process tool parts for tool call counts and timing."""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT json_extract(data, '$.tool') as tool_name, "
            "json_extract(data, '$.state.input.command') as command, "
            "json_extract(data, '$.state.time.start') as time_start, "
            "json_extract(data, '$.state.time.end') as time_end "
            "FROM part WHERE session_id = ? AND json_extract(data, '$.type') = 'tool'",
            (session_id,)
        )
        for row in cursor.fetchall():
            tool_name = row["tool_name"] or "unknown"
            stats.tool_calls[tool_name] = stats.tool_calls.get(tool_name, 0) + 1
            if row["time_start"] and row["time_end"]:
                stats.tool_time_ms += int(row["time_end"]) - int(row["time_start"])
            if tool_name == "bash" and row["command"]:
                stats.check_test_command(row["command"])

    @classmethod
    def _accumulate_type_time(cls, conn: sqlite3.Connection, session_id: str,
                              part_type: str, attr_name: str,
                              stats: "SessionStats") -> None:
        """Sum timing for a given part_type into the named attribute on stats."""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT json_extract(data, '$.time.start') as time_start, "
            "json_extract(data, '$.time.end') as time_end "
            "FROM part WHERE session_id = ? AND json_extract(data, '$.type') = ?",
            (session_id, part_type)
        )
        for row in cursor.fetchall():
            if row["time_start"] and row["time_end"]:
                setattr(stats, attr_name,
                        getattr(stats, attr_name) + int(row["time_end"]) - int(row["time_start"]))

    @classmethod
    def _query_database(cls, conn: sqlite3.Connection, session_id: str) -> "SessionStats":
        stats = cls._query_session_info(conn, session_id)
        cls._query_step_finish(conn, session_id, stats)
        cls._query_tool_parts(conn, session_id, stats)
        cls._accumulate_type_time(conn, session_id, "reasoning", "reasoning_time_ms", stats)
        cls._accumulate_type_time(conn, session_id, "text", "output_time_ms", stats)
        return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenCode Session Stats")
    parser.add_argument("session_id", help="Session ID or export file path")
    parser.add_argument("--db", help="Custom database path")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    session_path = Path(args.session_id)

    if session_path.exists() and session_path.suffix == ".json":
        stats = SessionStats.from_export_file(str(session_path))
    else:
        stats = SessionStats.from_database(args.session_id, args.db)

    if args.json:
        logger.info(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False))
    else:
        stats.print_report()