# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""轨迹分析器（trajectory-analyzer.ts 的 Python 移植版）。

核心：正则预处理 → agent 循环（5 工具，最多 50 轮）→ schema 校验 → 写 JSON。
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from trajectory_parser import (
    extract_key_sections,
    extract_skeleton,
    extract_skill_content,
    parse_stats,
    extract_gates,
    extract_errors,
)

REQUIRED_TOP_KEYS = [
    "sessionSummary",
    "workflowMeta",
    "sessionMeta",
    "flow",
    "skillQuality",
    "workflowLevelIssues",
    "optimizationPriorities",
]
REQUIRED_ARRAY_KEYS_IN_SESSION_META = ["cpsExecuted", "cpsMissing", "phasesNotReached"]

MAX_AGENT_ROUNDS = 50
MAX_READ_CALLS = 5

_TOOL_DESCRIPTIONS = """## 可用工具

每次回复调用一个工具，输出 JSON 格式：
{{"tool": "工具名", "args": {{...}}, "thought": "简述这步目的"}}

可用工具：
1. read — 读取文件指定行范围。args: {{"file": "路径", "line_start": N, "line_end": M}}
2. grep — 正则搜索文件。args: {{"pattern": "正则", "file": "路径", "head_limit": 200}}
3. bash — 执行 shell 命令。args: {{"command": "命令"}}
4. write_file — 写入/覆盖文件。args: {{"path": "路径", "content": "内容"}}
5. finish — 完成分析。args: {{}}

工作目录：{cwd}
轨迹文件：{traj_path}（通常数千行，不要试图一次读完）
输出文件：{analysis_path}

⚠️ 强制工作流（必须按此顺序执行）：

第 1 步：write_file — 写入 JSON 骨架到 {analysis_path}
  content = 7 个 key 的空骨架：sessionSummary(空串), workflowMeta(空对象),
  sessionMeta(cpsExecuted/cpsMissing/phasesNotReached 空数组), flow/skillQuality/
  workflowLevelIssues/optimizationPriorities(空数组)

第 2 步：write_file — 写填充脚本到 /tmp/fill_analysis.py
  脚本里用单引号（避免 JSON 转义），读取轨迹文件特定行，提取信息，更新 JSON

第 3 步：bash — 执行 python3 /tmp/fill_analysis.py

第 4 步：如需补充更多 key，重复第 2-3 步（每次写不同脚本填充不同 key）

第 5 步：bash — 验证 JSON：python3 -c 'import json; json.load(open("{analysis_path}")); print("valid")'

第 6 步：finish

⚠️ 禁止事项：
- 不要用 bash python3 -c "..."（JSON 转义复杂，容易出错）—— 改用 write_file 写脚本 + bash 执行
- 不要 grep 或 read 整个轨迹文件 — 初始上下文已包含完整骨架
- 前 3 轮内必须执行 write_file
- 需要看具体轨迹内容时，用 read 读取特定行段，每次不超过 200 行
- 不要在一次 write_file 中写入完整分析 JSON
"""


class AnalysisError(Exception):
    pass


class SchemaError(Exception):
    pass


@dataclass
class AIProviderConfig:
    base_url: str
    api_key: str
    model: str


@dataclass
class TrajectorySummary:
    skeleton: Any = None
    skill_content: Any = None
    stats: Any = None
    gates: Any = None
    errors: Any = None


def build_trajectory_summary(text: str) -> TrajectorySummary:
    return TrajectorySummary(
        skeleton=extract_skeleton(text),
        skill_content=extract_skill_content(text),
        stats=parse_stats(text),
        gates=extract_gates(text),
        errors=extract_errors(text),
    )


def validate_schema(obj: Any) -> None:
    if not isinstance(obj, dict):
        raise SchemaError("analysis output must be a JSON object")
    for key in REQUIRED_TOP_KEYS:
        if key not in obj:
            raise SchemaError(f"missing top-level key: {key}")
    for key in ("flow", "skillQuality", "workflowLevelIssues", "optimizationPriorities"):
        if not isinstance(obj.get(key), list):
            raise SchemaError(f"{key} must be an array")

    sm = obj.get("sessionMeta")
    if not isinstance(sm, dict) or isinstance(sm, list):
        raise SchemaError("sessionMeta must be an object")
    for key in REQUIRED_ARRAY_KEYS_IN_SESSION_META:
        if not isinstance(sm.get(key), list):
            raise SchemaError(f"sessionMeta.{key} must be an array")


@dataclass
class LlmResponse:
    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)
    duration_ms: int = 0
    reasoning: str = ""


def _llm_request(provider: AIProviderConfig, body: dict) -> LlmResponse:
    """HTTP 请求公共逻辑。body 已序列化为 dict（不含 system message 合并）。"""
    api_base = provider.base_url.rstrip("/")
    if api_base.endswith("/v1") or "/v1/" in api_base:
        chat_url = f"{api_base}/chat/completions"
    else:
        chat_url = f"{api_base}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            t0 = time.monotonic()
            resp = requests.post(chat_url, headers=headers, data=payload, timeout=600)
            duration_ms = int((time.monotonic() - t0) * 1000)
            if not resp.ok:
                raise AnalysisError(f"LLM API error: {resp.status_code} {resp.text}")
            data = resp.json()
            choices = data.get("choices") or []
            content = choices[0].get("message", {}).get("content") if choices else None
            reasoning = choices[0].get("message", {}).get("reasoning_content", "") if choices else ""
            if not content:
                raise AnalysisError("LLM API returned empty content")
            model = choices[0].get("message", {}).get("model") or data.get("model") or provider.model
            usage = data.get("usage") or {}
            return LlmResponse(content=content, model=model, usage=usage, duration_ms=duration_ms, reasoning=reasoning)
        except requests.exceptions.Timeout as e:
            raise AnalysisError(f"LLM 请求超时（600s）。可能上下文过大或模型负载高。") from e
        except Exception as e:
            last_err = e if isinstance(e, Exception) else AnalysisError(str(e))
            if attempt < 2:
                time.sleep(2)
                continue
    raise last_err or AnalysisError("LLM call failed")


def call_llm(provider: AIProviderConfig, system: str, messages: list[dict]) -> LlmResponse:
    """JSON 模式 LLM 调用（response_format=json_object，enable_thinking=True）。"""
    body = {
        "model": provider.model,
        "messages": [{"role": "system", "content": system}, *messages],
        "temperature": 0.3,
        "max_tokens": 16384,
        "response_format": {"type": "json_object"},
        "enable_thinking": True,
    }
    return _llm_request(provider, body)


def call_llm_text(provider: AIProviderConfig, system: str, user_text: str) -> str:
    """纯文本输出 LLM 调用（不加 response_format）。返回压缩后的纯文本。"""
    body = {
        "model": provider.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.3,
        "max_tokens": 16384,
    }
    return _llm_request(provider, body).content


def write_analysis_json(output_dir: str, trajectory_basename: str, obj: Any) -> str:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    out_path = p / f"{trajectory_basename}-analysis.json"
    out_path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return str(out_path)


@dataclass
class AnalysisResult:
    output_path: str
    steps: int
    analysis: Any


@dataclass
class RunContext:
    """管道运行的共享环境（输出位置 + 进度回调 + 日志器 + 分发模式）。"""
    output_dir: str
    output_basename: str
    on_progress: object = None
    logger: object = None
    mode: str = "agent"
    agent_io: Any = None


# ---- Agent 上下文 + 工具实现 ----

def _build_agent_context(summary: TrajectorySummary, key_sections: list,
                         traj_path: Path, analysis_path: Path,
                         line_count: int = 0) -> str:
    """初始 user 消息：骨架 + SKILL.md + stats + gates + errors + key_sections 摘要。"""
    sk = summary.skeleton
    skeleton_lines = [
        f"§{s.turn} {s.skill} ({s.type}) {s.status} @line {s.line}"
        for s in sk.skeleton[:100]
    ]
    sections_summary = [
        {"turn": s.turn, "skill": s.skill, "line": s.line}
        for s in key_sections
    ]
    gates_summary = [
        {"turn": g.turn, "result": g.result, "line": g.line}
        for g in summary.gates[:50]
    ]
    errors_summary = [
        {"turn": e.turn, "lineRange": list(e.line_range)}
        for e in summary.errors[:20]
    ]
    size_hint = f"（{line_count} 行，不要试图一次读完）" if line_count else "（通常数千行）"
    return "\n".join([
        "## 轨迹骨架",
        "\n".join(skeleton_lines),
        f"(共 {len(sk.skeleton)} calls / {sk.turn_count} turns)",
        "",
        "## Workflow SKILL.md 原文",
        summary.skill_content.skill_md or "(未找到 skill_content)",
        "",
        "## Stats",
        json.dumps(summary.stats, ensure_ascii=False),
        "",
        "## 门控结果",
        json.dumps(gates_summary, ensure_ascii=False),
        "",
        "## 异常段落",
        json.dumps(errors_summary, ensure_ascii=False),
        "",
        "## 关键片段（dispatch 子代理，仅 turn/skill/line 摘要）",
        json.dumps(sections_summary, ensure_ascii=False),
        "",
        f"轨迹文件：{traj_path} {size_hint}",
        f"输出文件：{analysis_path}",
        "",
        "⚠️ 初始上下文已包含完整骨架，不要重新 grep 或 read 整个文件。直接开始 write_file 写 JSON 骨架。",
    ])


def _tool_read(args: dict, traj_path: Path) -> str:
    file_path = args.get("file") or str(traj_path)
    line_start = int(args.get("line_start", 1))
    line_end = int(args.get("line_end", line_start + 100))
    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()
        start = max(0, line_start - 1)
        end = min(len(lines), line_end)
        content = "".join(lines[start:end])
        if len(content.encode("utf-8")) > 10240:
            content = content[:10240] + "\n... (truncated)"
        return content
    except Exception as e:
        return f"Error reading {file_path}: {e}"


def _tool_grep(args: dict, traj_path: Path) -> str:
    pattern = args.get("pattern", "")
    file_path = args.get("file") or str(traj_path)
    head_limit = int(args.get("head_limit", 200))
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex '{pattern}': {e}"
    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading {file_path}: {e}"
    matches = []
    for i, line in enumerate(lines, 1):
        if regex.search(line):
            matches.append(f"{i}: {line.rstrip()}")
            if len(matches) >= head_limit:
                break
    return "\n".join(matches) if matches else "(no matches)"


def _tool_bash(args: dict, output_dir: str) -> str:
    command = args.get("command", "")
    if not command.strip():
        return "(no output)"
    try:
        argv = shlex.split(command)
        result = subprocess.run(
            argv, shell=False,
            capture_output=True, text=True,
            timeout=60, cwd=str(output_dir),
        )
        output = result.stdout + result.stderr
        if len(output.encode("utf-8")) > 10240:
            output = output[:10240] + "\n... (truncated)"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out (60s)"
    except Exception as e:
        return f"Error executing command: {e}"


def _tool_write_file(args: dict) -> str:
    path = args.get("path", "")
    content = args.get("content", "")
    try:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        return f"OK, wrote {len(content.encode('utf-8'))} bytes to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


def _is_read_like_bash(command: str) -> bool:
    """bash 命令是否在读取文件内容（sed/cat/head/tail/grep/less）。"""
    for x in ("sed ", "cat ", "head ", "tail ", "grep ", "less ", "awk "):
        if x in command:
            return True
    return False


def _truncate_tool_results(messages: list[dict], keep_last: int = 3,
                           max_old_len: int = 500) -> None:
    """截断旧的 tool result 消息，只保留最近 keep_last 个完整结果。"""
    tool_result_indices = []
    for i, m in enumerate(messages):
        if (m["role"] == "user" and isinstance(m["content"], str)
                and "result]" in m["content"][:30]):
            tool_result_indices.append(i)
    if len(tool_result_indices) > keep_last:
        for idx in tool_result_indices[:-keep_last]:
            msg = messages[idx]
            if len(msg["content"]) > max_old_len:
                msg["content"] = msg["content"][:max_old_len] + "\n... (truncated to save context)"


def _execute_tool(tool: str, args: dict, traj_path: Path,
                  analysis_path: Path, output_dir: str) -> str:
    if tool == "read":
        return _tool_read(args, traj_path)
    if tool == "grep":
        return _tool_grep(args, traj_path)
    if tool == "bash":
        return _tool_bash(args, output_dir)
    if tool == "write_file":
        return _tool_write_file(args)
    if tool == "finish":
        return "(finish)"
    return f"Error: unknown tool '{tool}'"


# ---- Agent 循环 ----

def _log_skeleton(summary: TrajectorySummary, on_progress, logger) -> None:
    msg = f"骨架：{len(summary.skeleton.skeleton)} calls / {summary.skeleton.turn_count} turns"
    detail = {"gates": len(summary.gates), "errors": len(summary.errors)}
    if on_progress:
        on_progress({"stage": "skeleton", "msg": msg, "detail": detail})
    if logger:
        logger.log_system("progress", {
            "stage": "skeleton", "msg": msg,
            "detail": {**detail, "skillName": summary.skeleton.skeleton[0].skill
                       if summary.skeleton.skeleton else ""}})


def _log_key_sections(key_sections: list, on_progress, logger) -> None:
    msg = f"提取 {len(key_sections)} 个关键片段"
    if on_progress:
        on_progress({"stage": "key-sections", "msg": msg})
    if logger:
        logger.log_system("progress", {"stage": "key-sections", "msg": msg})


def _setup_agent_run(summary: TrajectorySummary, key_sections: list,
                     trajectory_text: str, prompt_md: str,
                     ctx: RunContext) -> tuple[Path, Path, str, list[dict]]:
    traj_path = Path(ctx.output_dir) / f"{ctx.output_basename}.md"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_text(trajectory_text, encoding="utf-8")
    analysis_path = Path(ctx.output_dir) / f"{ctx.output_basename}-analysis.json"
    system = prompt_md + "\n\n" + _TOOL_DESCRIPTIONS.format(
        cwd=ctx.output_dir, traj_path=str(traj_path), analysis_path=str(analysis_path))
    line_count = len(trajectory_text.split("\n"))
    initial_msg = _build_agent_context(summary, key_sections, traj_path, analysis_path, line_count)
    messages: list[dict] = [{"role": "user", "content": initial_msg}]
    if ctx.logger:
        ctx.logger.log_system("prompt", {"system": system, "length": len(system)})
        ctx.logger.log_user_text(initial_msg)
    return traj_path, analysis_path, system, messages


def _log_assistant_step(logger, resp: LlmResponse) -> None:
    if not logger:
        return
    blocks = []
    if resp.reasoning:
        blocks.append({"type": "thinking", "text": resp.reasoning})
    blocks.append({"type": "text", "text": resp.content})
    logger.log_assistant(blocks, model=resp.model,
                         usage=_normalize_usage(resp.usage), duration_ms=resp.duration_ms)


def _count_tool_access(tool: str, args: dict, read_count: int,
                       write_count: int) -> tuple[int, int]:
    if tool in ("read", "grep"):
        return read_count + 1, write_count
    if tool == "bash" and _is_read_like_bash(args.get("command", "")):
        return read_count + 1, write_count
    if tool == "write_file":
        return read_count, write_count + 1
    return read_count, write_count


def _balance_hint(read_count: int, write_count: int, tool: str, args: dict):
    is_read_like = tool in ("read", "grep") or (
        tool == "bash" and _is_read_like_bash(args.get("command", "")))
    if not is_read_like:
        return None
    if read_count >= MAX_READ_CALLS:
        return (f"⛔ 已达到读取上限（{MAX_READ_CALLS} 次）。"
                f"请立即用 bash python3 -c 填充剩余 key（sessionSummary, workflowMeta, "
                f"sessionMeta, skillQuality, workflowLevelIssues, optimizationPriorities），"
                f"然后验证并 finish。不要再 read/grep。")
    if write_count >= 1 and read_count >= 3:
        return ("骨架已写入。请用 bash python3 -c 填充剩余 key，不要再 read 轨迹文件。"
                "初始上下文的骨架+SKILL.md 已有足够信息。")
    if read_count >= 3 and write_count == 0:
        return (f"⚠️ 已读 {read_count} 次但未开始写。请立即用 write_file 写 JSON 骨架"
                f"（7 个 key，数组值为空 []），然后用 bash python3 -c 逐步填充。")
    return None


def _start_heartbeat(round_num: int, on_progress, logger) -> threading.Event:
    stop = threading.Event()

    def heartbeat(r=round_num, _stop=stop):
        secs = 0
        while not _stop.wait(20):
            secs += 20
            if on_progress:
                on_progress({"stage": "ping", "round": r, "msg": f"等待 LLM 响应… {secs}s"})
            if logger:
                logger.log_system("progress", {"stage": "ping", "round": r,
                    "msg": f"等待 LLM 响应… {secs}s"})
    threading.Thread(target=heartbeat, daemon=True).start()
    return stop


@dataclass
class _AgentLoopState:
    messages: list
    traj_path: Path
    analysis_path: Path
    output_dir: str
    on_progress: object = field(default=None)
    logger: object = field(default=None)
    read_count: int = 0
    write_count: int = 0


def _append_turn_messages(tool: str, resp_content: str, result: str,
                            state: _AgentLoopState, args: dict) -> None:
    state.messages.append({"role": "assistant", "content": resp_content})
    tool_result_msg = f"[{tool} result]: {result}"
    state.messages.append({"role": "user", "content": tool_result_msg})
    if state.logger:
        state.logger.log_user_text(tool_result_msg)
    hint = _balance_hint(state.read_count, state.write_count, tool, args)
    if hint:
        state.messages.append({"role": "user", "content": hint})
        if state.logger:
            state.logger.log_user_text(hint)


def _process_llm_response(resp: LlmResponse, round_num: int,
                            state: _AgentLoopState) -> bool:
    try:
        tool_call = json.loads(resp.content)
        if isinstance(tool_call, list) and tool_call:
            tool_call = tool_call[0]
        tool = tool_call["tool"]
        args = tool_call.get("args", {}) or {}
        thought = tool_call.get("thought", "")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        err = (f"输出不是有效的工具调用 JSON：{e}。"
               f'请输出格式：{{"tool":"...","args":{{...}},"thought":"..."}}')
        state.messages.append({"role": "assistant", "content": resp.content})
        state.messages.append({"role": "user", "content": err})
        _log_assistant_step(state.logger, resp)
        if state.logger:
            state.logger.log_user_text(err)
        return False
    _log_assistant_step(state.logger, resp)
    if state.on_progress:
        state.on_progress({"stage": "agent", "round": round_num, "msg": f"{tool}: {thought[:80]}"})
    result = _execute_tool(tool, args, state.traj_path, state.analysis_path, state.output_dir)
    state.read_count, state.write_count = _count_tool_access(
        tool, args, state.read_count, state.write_count)
    if tool == "finish":
        return True
    _append_turn_messages(tool, resp.content, result, state, args)
    return False


def _run_agent_loop(provider: AIProviderConfig, system: str,
                    state: _AgentLoopState) -> int:
    round_num = 0
    for round_num in range(1, MAX_AGENT_ROUNDS + 1):
        if state.on_progress:
            state.on_progress({"stage": "agent", "round": round_num, "msg": "等待 LLM 响应…"})
        stop = _start_heartbeat(round_num, state.on_progress, state.logger)
        try:
            _truncate_tool_results(state.messages)
            resp = call_llm(provider, system, state.messages)
        finally:
            stop.set()
        if _process_llm_response(resp, round_num, state):
            break
    return round_num


def _finalize_agent(analysis_path: Path, round_num: int,
                     ctx: RunContext) -> AnalysisResult:
    if not analysis_path.exists():
        raise AnalysisError(f"分析完成但输出文件不存在：{analysis_path}")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    validate_schema(analysis)
    if ctx.on_progress:
        ctx.on_progress({"stage": "done", "msg": f"完成（{round_num} 轮）"})
    if ctx.logger:
        ctx.logger.log_result(f"完成（{round_num} 轮）", subtype="success")
        ctx.logger.close()
    return AnalysisResult(output_path=str(analysis_path), steps=round_num, analysis=analysis)


def run_agent_pipeline(
    trajectory_text: str,
    prompt_md: str,
    provider: AIProviderConfig,
    ctx: RunContext,
) -> AnalysisResult:
    try:
        # Step 0：骨架 + 上下文提取
        summary = build_trajectory_summary(trajectory_text)
        _log_skeleton(summary, ctx.on_progress, ctx.logger)
        key_sections = extract_key_sections(trajectory_text, summary.skeleton)
        _log_key_sections(key_sections, ctx.on_progress, ctx.logger)

        traj_path, analysis_path, system, messages = _setup_agent_run(
            summary, key_sections, trajectory_text, prompt_md, ctx)

        state = _AgentLoopState(
            messages=messages, traj_path=traj_path, analysis_path=analysis_path,
            output_dir=ctx.output_dir, on_progress=ctx.on_progress, logger=ctx.logger)
        round_num = _run_agent_loop(provider, system, state)

        return _finalize_agent(analysis_path, round_num, ctx)
    except Exception as e:
        if ctx.logger:
            ctx.logger.log_result(f"失败：{type(e).__name__}: {e}", subtype="error")
            ctx.logger.close()
        raise


_FIX_SENTINEL = object()


def _try_load_candidate(candidate: str, pos: int, progressed):
    """尝试解析 candidate。成功返回 (value, progressed)；失败返回 (SENTINEL, 可能更新的 progressed)。"""
    try:
        return json.loads(candidate), progressed
    except json.JSONDecodeError as e2:
        if e2.pos > pos and progressed is None:
            progressed = candidate
        return _FIX_SENTINEL, progressed


def _scan_insertion_fixes(current: str, pos: int, progressed):
    """尝试在 pos 附近插入 } / ] 等括号修复。"""
    for ins in ("}", "]", "}}", "]]", "}]"):
        for offset in (0, -1, 1, -2, 2):
            p = max(0, min(len(current), pos + offset))
            cand = current[:p] + ins + current[p:]
            parsed, progressed = _try_load_candidate(cand, pos, progressed)
            if parsed is not _FIX_SENTINEL:
                return parsed, progressed
    return _FIX_SENTINEL, progressed


def _try_deletion_fix(current: str, pos: int, progressed):
    """尝试删除错误点的 { } , ]（修多余/错位的括号）。"""
    if pos < len(current) and current[pos] in "{},]":
        cand = current[:pos] + current[pos + 1:]
        return _try_load_candidate(cand, pos, progressed)
    return _FIX_SENTINEL, progressed


def _attempt_one_fix(current: str, err: json.JSONDecodeError):
    """对一次解析失败尝试所有修复。返回 (parsed_or_SENTINEL, progressed_candidate_or_None)。"""
    pos = err.pos
    progressed = None
    parsed, progressed = _scan_insertion_fixes(current, pos, progressed)
    if parsed is not _FIX_SENTINEL:
        return parsed, None
    parsed, progressed = _try_deletion_fix(current, pos, progressed)
    if parsed is not _FIX_SENTINEL:
        return parsed, None
    return _FIX_SENTINEL, progressed


class _Retry(Exception):
    """_load_json_lenient 内部控制流：用修复后进度更大的候选重试。"""
    def __init__(self, current: str):
        super().__init__(current)
        self.current = current


def _load_or_fix(current: str):
    """成功返回解析值；否则抛 _Retry(下轮 current) 或原 JSONDecodeError。"""
    try:
        return json.loads(current)
    except json.JSONDecodeError as err:
        # 对 JSONDecodeError 做修复尝试；无法修复则外抛由调用者处理
        parsed, progressed = _attempt_one_fix(current, err)
        if parsed is not _FIX_SENTINEL:
            return parsed
        if progressed is None:
            raise
        raise _Retry(progressed) from err


def _load_json_lenient(text: str, max_fixes: int = 50) -> Any:
    """解析 JSON；失败时迭代修复 claude code 偶发的结构错误（自校验时序问题导致的残留畸形）。

    自校验 prompt 指令让 claude code 写完 json.load 验，但 Write 工具的时序问题
    导致校验有时假通过、磁盘残留中间畸形版。此函数作下游兜底，迭代修复：
    - 插入 } / ]（修漏括号）
    - 删除错误点的 { } , ]（修多余/错位的括号）
    每轮优先取能整段解析的；否则取使错误位置后移（有进展）的，继续下一轮。
    """
    current = text
    for _ in range(max_fixes):
        try:
            return _load_or_fix(current)
        except _Retry as r:
            current = r.current
    return json.loads(current)


CLAUDE_CODE_TIMEOUT_SEC = 1500  # 25 分钟（确定性压缩省了 LLM 压缩时间，预算允许 claude code 多跑）


def _tool_use_detail(block: dict) -> str:
    name = block.get("name", "")
    inp = block.get("input", {}) or {}
    if "file_path" in inp:
        detail = inp.get("file_path", "")
    elif "command" in inp:
        detail = inp.get("command", "")[:80]
    elif "path" in inp:
        detail = inp.get("path", "")
    elif "pattern" in inp:
        detail = f"pattern={inp.get('pattern','')[:60]}"
    else:
        detail = json.dumps(inp, ensure_ascii=False)[:80]
    return f"{name}: {detail}"


def _assistant_block_detail(block) -> tuple[str, str] | None:
    if not isinstance(block, dict):
        return None
    btype = block.get("type")
    if btype == "tool_use":
        return ("tool", _tool_use_detail(block))
    if btype == "text":
        return ("text", block.get("text", "")[:120])
    if btype == "thinking":
        return ("thinking", block.get("thinking", "")[:120])
    return None


def _parse_claude_stream_event(evt: dict) -> tuple[str, str]:
    """从 claude stream-json 事件提取 (stage_label, msg) 用于进度展示。"""
    etype = evt.get("type")
    if etype == "system":
        return ("init", f"claude code 初始化（model={evt.get('model','?')}）")
    if etype == "assistant":
        msg = evt.get("message", {})
        for block in msg.get("content", []):
            detail = _assistant_block_detail(block)
            if detail is not None:
                return detail
        return ("assistant", "")
    if etype == "user":
        return ("tool-result", "")
    if etype == "result":
        sub = evt.get("subtype", "")
        turns = evt.get("num_turns", 0)
        cost = evt.get("total_cost_usd", 0)
        return ("result", f"claude code {sub}（{turns} turns, ${cost:.4f}）")
    return (etype or "unknown", "")


def _prepare_claude_run(trajectory_text: str, prompt_md: str,
                         ctx: RunContext):
    out_dir = Path(ctx.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = (out_dir / f"{ctx.output_basename}.md").resolve()
    traj_path.write_text(trajectory_text, encoding="utf-8")
    analysis_path = (out_dir / f"{ctx.output_basename}-analysis.json").resolve()
    # 清理旧的分析文件，避免读到上次残留
    if analysis_path.exists():
        analysis_path.unlink()
    line_count = len(trajectory_text.split("\n"))
    if ctx.on_progress:
        ctx.on_progress({"stage": "claude-start",
                         "msg": f"启动 claude code（轨迹 {line_count} 行）"})
    if ctx.logger:
        ctx.logger.log_system("progress", {"stage": "claude-start",
            "msg": f"启动 claude code（轨迹 {line_count} 行）"})
    user_instruction = (
        f"{prompt_md}\n\n"
        f"---\n\n"
        f"请分析轨迹文件 {traj_path}。\n"
        f"轨迹文件 {line_count} 行——不要一次读完，用 Grep 提取骨架后按需 Read 段落。\n"
        f"将分析结果 JSON 用 Write 工具写入 {analysis_path}（以此绝对路径为准，"
        f"不要改写到 logs/ 或其他目录）。\n"
        f"写完即回复 DONE，不要自行跑脚本校验或做其他操作（服务端会做 JSON 修复）。"
    )
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    cmd = [
        "claude", "-p", user_instruction,
        "--output-format", "stream-json",
        "--verbose",
        "--allowedTools", "Read Write Edit Bash Grep Glob",
        "--permission-mode", "acceptEdits",
    ]
    if ctx.logger:
        ctx.logger.log_system("prompt", {"user_len": len(user_instruction),
            "prompt_md_len": len(prompt_md), "mode": "user-message",
            "traj_path": str(traj_path), "analysis_path": str(analysis_path)})
    return traj_path, analysis_path, env, cmd


def _spawn_claude(cmd: list, env: dict):
    try:
        return subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env,
        )
    except FileNotFoundError as e:
        raise AnalysisError(f"未找到 claude CLI：{e}。请确认已安装并位于 PATH。") from e


def _classify_claude_line(raw_line: str):
    """返回 (stage_label, msg) 或 None（空行/解析失败/无消息时跳过）。"""
    raw_line = raw_line.strip()
    if not raw_line:
        return None
    try:
        evt = json.loads(raw_line)
    except json.JSONDecodeError:
        return None
    stage_label, msg = _parse_claude_stream_event(evt)
    if not msg:
        return None
    return stage_label, msg


def _emit_claude_event(stage_label: str, msg: str, turn_count: list,
                        on_progress, logger) -> None:
    if stage_label in ("tool", "text", "thinking"):
        turn_count[0] += 1
        if on_progress:
            on_progress({"stage": "claude", "round": turn_count[0], "msg": msg})
        if logger and stage_label == "tool":
            logger.log_system("progress", {"stage": "claude-tool",
                "round": turn_count[0], "msg": msg})
    elif stage_label == "result":
        if on_progress:
            on_progress({"stage": "claude-done", "msg": msg})
        if logger:
            logger.log_system("progress", {"stage": "claude-done", "msg": msg})


def _stream_claude(proc, turn_count: list, timer: threading.Timer,
                    on_progress, logger) -> None:
    # 心跳：claude code 生成大 JSON 期间长时间无 tool 事件，需定期发 ping
    # 保持 Python→Next.js 连接活跃，避免 "Error in input stream"（连接空闲超时）
    heartbeat_stop = threading.Event()

    def _heartbeat():
        secs = 0
        while not heartbeat_stop.wait(20):
            secs += 20
            if on_progress:
                with suppress(Exception):
                    on_progress({"stage": "ping", "round": turn_count[0],
                                 "msg": f"claude code 生成中… {secs}s"})
    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()
    try:
        for raw_line in proc.stdout:
            cls = _classify_claude_line(raw_line)
            if cls is None:
                continue
            _emit_claude_event(cls[0], cls[1], turn_count, on_progress, logger)
    finally:
        heartbeat_stop.set()
        hb_thread.join(timeout=2)
        timer.cancel()
        proc.stdout.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def _collect_claude_result(analysis_path: Path, turn_count: int,
                            exit_info: tuple, ctx: RunContext) -> AnalysisResult:
    # 即使超时/异常退出，只要 claude 已写出 analysis.json 就先尝试读取+修复；
    # 能用则直接返回，避免因 claude 在收尾阶段超时而丢弃已产出的有效分析。
    timed_out, returncode, stderr_output = exit_info
    analysis = None
    analysis_parse_err = None
    if analysis_path.exists():
        try:
            analysis = _load_json_lenient(analysis_path.read_text(encoding="utf-8"))
            validate_schema(analysis)
        except Exception as e:
            analysis_parse_err = e
            analysis = None

    if analysis is not None:
        steps = turn_count
        if ctx.on_progress:
            ctx.on_progress({"stage": "done", "msg": f"完成（{steps} turns）"})
        if ctx.logger:
            ctx.logger.log_result(f"完成（{steps} turns）", subtype="success")
            ctx.logger.close()
        return AnalysisResult(output_path=str(analysis_path), steps=steps, analysis=analysis)

    # 无可用 JSON —— 报具体错误
    if timed_out:
        raise AnalysisError(
            f"claude code 超时（{CLAUDE_CODE_TIMEOUT_SEC}s）且未产出可用 JSON。"
            f"stderr: {stderr_output[:500]}")
    if returncode not in (0, None):
        raise AnalysisError(
            f"claude code 退出码 {returncode} 且未产出可用 JSON。stderr: {stderr_output[:500]}")
    if analysis_parse_err is not None:
        raise AnalysisError(f"analysis.json 解析失败：{analysis_parse_err}")
    raise AnalysisError(
        f"claude code 未产出 analysis.json：{analysis_path}。stderr: {stderr_output[:500]}")


def run_claude_code_pipeline(
    trajectory_text: str,
    prompt_md: str,
    provider: AIProviderConfig | None,
    ctx: RunContext,
) -> AnalysisResult:
    """v3：直接调用本机 claude code CLI 分析轨迹。

    将 prompts/session-trajectory-analyse.md 作为用户消息的一部分（非 --append-system-prompt，
    后者会叠加在 claude code 自带 system prompt 上导致 glm-5.2 首次思考卡死），
    让 claude code 用 Read/Grep/Bash/Write 工具分析轨迹文件，
    最终输出 JSON 到 analysis_path。超时 15 分钟。
    """
    try:
        traj_path, analysis_path, env, cmd = _prepare_claude_run(
            trajectory_text, prompt_md, ctx)
        proc = _spawn_claude(cmd, env)

        timed_out = {"flag": False}

        def _kill():
            timed_out["flag"] = True
            proc.kill()

        timer = threading.Timer(CLAUDE_CODE_TIMEOUT_SEC, _kill)
        timer.start()

        turn_count = [0]
        _stream_claude(proc, turn_count, timer, ctx.on_progress, ctx.logger)

        stderr_output = ""
        with suppress(Exception):
            stderr_output = proc.stderr.read() if proc.stderr else ""

        return _collect_claude_result(
            analysis_path, turn_count[0],
            (timed_out["flag"], proc.returncode, stderr_output), ctx)
    except Exception as e:
        if ctx.logger:
            ctx.logger.log_result(f"失败：{type(e).__name__}: {e}", subtype="error")
            ctx.logger.close()
        raise


def _efficiency_rating(env: dict) -> str:
    """从确定性信封算 efficiency.rating（不依赖 LLM）。

    阈值是初步启发式：错误多/重试多/单 agent turn 数过大→weak/fail。
    """
    error_count = env.get("errorCount", 0)
    retry_count = env.get("retryCount", 0)
    turn_count = env.get("turnCount", 0)
    if error_count >= 5 or retry_count >= 5 or turn_count >= 80:
        return "fail"
    if error_count >= 2 or retry_count >= 2 or turn_count >= 30:
        return "weak"
    return "pass"


def validate_v4_schema(obj: Any) -> None:
    """v4 LLM 输出最小校验：sessionSummary + agents[]，每 agent 有 id + dimensions.completion/quality.rating。"""
    if not isinstance(obj, dict):
        raise SchemaError("v4 analysis output must be a JSON object")
    if not isinstance(obj.get("sessionSummary"), str):
        raise SchemaError("sessionSummary must be a string")
    agents = obj.get("agents")
    if not isinstance(agents, list):
        raise SchemaError("agents must be an array")
    if len(agents) == 0:
        raise SchemaError("agents must not be empty")
    for a in agents:
        if not isinstance(a, dict) or not a.get("id"):
            raise SchemaError("each agent needs an id")
        dims = a.get("dimensions")
        if not isinstance(dims, dict):
            raise SchemaError(f"agent {a.get('id')} missing dimensions")
        for k in ("completion", "quality"):
            d = dims.get(k)
            if not isinstance(d, dict) or d.get("rating") not in ("pass", "weak", "fail", "n-a"):
                raise SchemaError(f"agent {a.get('id')} dimensions.{k}.rating invalid")


def _synthesize_evidence(io_a: dict, dim_key: str) -> str:
    """LLM 漏 evidence 时，用确定性 envelope/actions 合成一条依据，保证可追溯。"""
    env = io_a.get("envelope", {}) or {}
    artifacts = io_a.get("artifacts", []) or []
    actions = io_a.get("actions", []) or []
    err_acts = [a for a in actions if a.get("state") == "error"]
    parts = [
        f"envelope: turn={env.get('turnCount',0)}, err={env.get('errorCount',0)}, retry={env.get('retryCount',0)}",
        f"artifacts={len(artifacts)}",
        f"actions={len(actions)}({len(err_acts)}错)",
    ]
    if artifacts:
        parts.append("产出:" + ", ".join(a.split(': ', 1)[-1] for a in artifacts[:3]))
    if dim_key == "completion":
        return "（合成依据）" + "; ".join(parts)
    return "（合成依据）" + "; ".join(parts)  # quality 同样基于这些客观信号


def inject_v4_merge(analysis: dict, agent_io: dict) -> dict:
    """用确定性 agent_io 覆盖/补全 LLM 产出的结构字段（envelope/name/input/output/artifacts/parentId/
    role），并填 efficiency.rating（LLM 的 efficiency.note/diagnosis/suggestion 若有则保留）。
    保证每个 agent_io 里的 agent 都出现在结果中（LLM 漏的补 n-a stub）。
    LLM 漏 evidence 时，用 envelope/actions 合成依据兜底（保证每个 rating 都可追溯）。
    """
    io_agents = agent_io.get("agents", []) if isinstance(agent_io, dict) else []
    by_id = {a.get("id"): a for a in io_agents if isinstance(a, dict) and a.get("id")}
    llm_by_id = {a.get("id"): a for a in analysis.get("agents", []) if isinstance(a, dict) and a.get("id")}

    merged_agents = []
    for io_a in io_agents:
        aid = io_a.get("id")
        llm_a = llm_by_id.get(aid, {})
        llm_dims = llm_a.get("dimensions", {}) if isinstance(llm_a.get("dimensions"), dict) else {}
        comp_raw = llm_dims.get("completion")
        comp = comp_raw if isinstance(comp_raw, dict) else {"rating": "n-a", "note": ""}
        qual_raw = llm_dims.get("quality")
        qual = qual_raw if isinstance(qual_raw, dict) else {"rating": "n-a", "note": ""}
        eff_llm = llm_dims.get("efficiency") if isinstance(llm_dims.get("efficiency"), dict) else {}
        env = io_a.get("envelope", {})
        # 兜底：LLM 漏 evidence 时合成一条确定性依据
        if not comp.get("evidence"):
            comp = {**comp, "evidence": _synthesize_evidence(io_a, "completion")}
        if not qual.get("evidence"):
            qual = {**qual, "evidence": _synthesize_evidence(io_a, "quality")}
        eff = {
            "rating": _efficiency_rating(env),
            "note": eff_llm.get("note", ""),
            "evidence": eff_llm.get("evidence") or _synthesize_evidence(io_a, "efficiency"),
            "diagnosis": eff_llm.get("diagnosis"),
            "suggestion": eff_llm.get("suggestion"),
        }
        merged_agents.append({
            "id": aid,
            "parentId": io_a.get("parentId"),
            "role": io_a.get("role"),
            "name": io_a.get("name"),
            "inputSummary": io_a.get("inputSummary", ""),
            "outputSummary": io_a.get("outputSummary", ""),
            "artifacts": io_a.get("artifacts", []),
            "actions": io_a.get("actions", []),
            "turns": io_a.get("turns", []),
            "envelope": env,
            "dimensions": {"completion": comp, "efficiency": eff, "quality": qual},
        })
    analysis["agents"] = merged_agents
    return analysis


def _prepare_v4_paths(ctx: RunContext, agent_io: Any):
    out_dir = Path(ctx.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agentio_path = (out_dir / f"{ctx.output_basename}-agentio.json").resolve()
    agentio_path.write_text(
        json.dumps(agent_io, ensure_ascii=False, indent=2), encoding="utf-8")
    analysis_path = (out_dir / f"{ctx.output_basename}-analysis.json").resolve()
    if analysis_path.exists():
        analysis_path.unlink()
    return agentio_path, analysis_path


def _validate_v4_inputs(agent_io: Any, provider):
    if not isinstance(agent_io, dict):
        raise AnalysisError("agent_io 不是对象")
    agents = agent_io.get("agents", [])
    if provider is None:
        raise AnalysisError("v4 需要 provider 配置")
    if not agents:
        raise AnalysisError("agent_io.agents 为空")
    return agents, agent_io.get("taskQuery", ""), len(agents)


def _run_per_agent_audit(agents: list, provider: AIProviderConfig,
                          ctx: RunContext) -> dict:
    per_agent: dict[str, dict] = {}
    done = {"n": 0}
    agent_count = len(agents)

    def _audit_one(agent: dict) -> tuple[str, dict]:
        return agent.get("id", "?"), _audit_one_agent(provider, agent)

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_audit_one, a): a for a in agents}
        for fut in as_completed(futs):
            aid, dims = fut.result()
            per_agent[aid] = dims
            done["n"] += 1
            if ctx.on_progress:
                ctx.on_progress({"stage": "claude", "round": done["n"],
                             "msg": f"已审计 {done['n']}/{agent_count}: {aid}"})
            if ctx.logger:
                ctx.logger.log_system("progress", {"stage": "claude-agent",
                    "round": done["n"], "id": aid,
                    "completion": dims.get("completion", {}).get("rating"),
                    "quality": dims.get("quality", {}).get("rating")})
    return per_agent


def _build_v4_summary_list(agents: list, per_agent: dict) -> list:
    summary_list = []
    for a in agents:
        aid = a.get("id", "?")
        d = per_agent.get(aid, {})
        summary_list.append({
            "id": aid, "role": a.get("role"), "name": a.get("name"),
            "completion": (d.get("completion") or {}).get("rating", "n-a"),
            "quality": (d.get("quality") or {}).get("rating", "n-a"),
            "note": (d.get("completion") or {}).get("note", ""),
        })
    return summary_list


def _assemble_v4_analysis(agents: list, per_agent: dict, agg: dict,
                           agent_io: Any) -> dict:
    llm_analysis = {
        "sessionSummary": agg.get("sessionSummary", ""),
        "agents": [
            {"id": a.get("id"),
             "dimensions": {
                 "completion": per_agent.get(a.get("id"), {}).get("completion", {"rating": "n-a", "note": ""}),
                 "quality": per_agent.get(a.get("id"), {}).get("quality", {"rating": "n-a", "note": ""}),
                 "efficiency": per_agent.get(a.get("id"), {}).get("efficiency", {}),
             }}
            for a in agents
        ],
        "crossIssues": agg.get("crossIssues", []),
        "optimizationPriorities": agg.get("optimizationPriorities", []),
    }
    validate_v4_schema(llm_analysis)
    return inject_v4_merge(llm_analysis, agent_io)


def _v4_progress(on_progress, logger, stage: str, prog_msg: str, log_msg: str) -> None:
    if on_progress:
        on_progress({"stage": stage, "msg": prog_msg})
    if logger:
        logger.log_system("progress", {"stage": stage, "msg": log_msg})


def run_v4_pipeline(
    agent_io: Any,
    prompt_md: str,
    ctx: RunContext,
    provider: AIProviderConfig | None = None,
) -> AnalysisResult:
    """v4：agent 中心三维度审计（并行版）。agent-IO 由 Next.js 路由确定性算好传入。

    流程：
    1. 写 agentio.json（留档）。
    2. 并行 per-agent LLM 调用（ThreadPoolExecutor, max 6）：每个 agent 一次 json_object 调用，
       输入只含该 agent 的 input/output/actions/artifacts/envelope → {completion, quality, efficiency:{note?}}。
       每 agent 输入小、不截断、覆盖 100%；互不依赖天然并行。
    3. 一次聚合 LLM 调用：所有 agent 的 ratings 摘要 + taskQuery → sessionSummary / crossIssues / optimizationPriorities。
    4. 组装 LLM-analysis 形态 → validate_v4_schema → inject_v4_merge（用 agent_io 填结构 + 确定性 efficiency.rating）。
    """
    try:
        agentio_path, analysis_path = _prepare_v4_paths(ctx, agent_io)
        agents, task_query, agent_count = _validate_v4_inputs(agent_io, provider)

        _v4_progress(ctx.on_progress, ctx.logger, "claude-start",
                     f"并行审计 {agent_count} 个 agent（每 agent 一次 LLM 调用）…",
                     f"v4 并行审计 {agent_count} 个 agent")

        per_agent = _run_per_agent_audit(agents, provider, ctx)

        _v4_progress(ctx.on_progress, ctx.logger, "claude-done",
                     f"per-agent 完成（{agent_count}），开始聚合…",
                     f"per-agent 完成（{agent_count}）")

        agg = _aggregate_session(
            provider, task_query, _build_v4_summary_list(agents, per_agent))
        analysis = _assemble_v4_analysis(agents, per_agent, agg, agent_io)

        analysis_path.write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        steps = agent_count
        if ctx.on_progress:
            ctx.on_progress({"stage": "done", "msg": f"完成（{steps} 个 agent）"})
        if ctx.logger:
            ctx.logger.log_result(f"完成（{steps} 个 agent）", subtype="success")
            ctx.logger.close()
        return AnalysisResult(output_path=str(analysis_path), steps=steps, analysis=analysis)
    except Exception as e:
        if ctx.logger:
            ctx.logger.log_result(f"失败：{type(e).__name__}: {e}", subtype="error")
            ctx.logger.close()
        raise


V4_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")


def _load_v4_prompt(filename: str, fallback: str) -> str:
    """调用时从 prompts/ 读 .md（热更新，编辑后下次 v4 即生效）；读不到回退内联。"""
    try:
        return Path(os.path.join(V4_PROMPTS_DIR, filename)).read_text(encoding="utf-8")
    except Exception:
        return fallback


V4_AGENT_SYSTEM = """你是 CANNBot Agent 审计顾问，对**单个 agent** 做三维度评判。输入是该 agent 的 JSON。

维度：
- completion（功能完成情况）：input 意图 vs actions/output/artifacts。**核心产出未交付→fail；已交付但有瑕疵/重试后达成→weak；覆盖意图且无遗留 error→pass；证据不足→n-a**。对主 agent（编排者）看整体交付：哪怕部分子任务已勾选，整体交付物未完成→fail。
- quality（开发质量）：派发标准 vs actions（读了什么/改了什么/verifier 结果）+ artifacts + error/retry。pass/weak/fail/n-a。
- efficiency（开发效率）：**rating 留空字符串 ""**（服务端用 envelope 确定性填），只在有明显效率问题（串行可并行/重复读/重试浪费/turn 过多）时写 note/diagnosis/suggestion，否则给空对象。

纪律：
- rating 必须与结论一致：结论是"未交付/中途终止"→ 必须 fail，不得给 weak。
- **不要只看 outputSummary 自评"已完成"**，必须对照 actions 实际动作判。
- diagnosis 必须以根因前缀开头：[skill-defect]/[execution-deviation]/[infra-issue]/[workflow-design]。
- pass/n-a 只写 rating+note；weak/fail 必填 evidence+diagnosis+suggestion。
- evidence 引用 #turn tool 或 envelope/artifacts。

输出严格 JSON：{"completion":{"rating":"","note":"",...},"quality":{"rating":"","note":"",...},"efficiency":{"note":""}}
"""


def _audit_one_agent(provider: AIProviderConfig, agent: dict) -> dict:
    """对单个 agent 调一次 LLM，返回 {completion, quality, efficiency}。失败回退全 n-a。"""
    user = (
        "请审计下面这个 agent 的三维度，输出严格 JSON。\n"
        "agent 数据（JSON）：\n"
        f"{json.dumps(agent, ensure_ascii=False)}"
    )
    try:
        resp = call_llm(
            provider,
            _load_v4_prompt("audit-v4-agent.md", V4_AGENT_SYSTEM),
            [{"role": "user", "content": user}],
        )
        data = _load_json_lenient(resp.content)
        if not isinstance(data, dict):
            raise AnalysisError("LLM 返回非 JSON 对象")
        dims = data.get("dimensions", data) if isinstance(data.get("dimensions"), dict) else data
        completion = dims.get("completion") if isinstance(dims.get("completion"), dict) else {}
        quality = dims.get("quality") if isinstance(dims.get("quality"), dict) else {}
        eff = dims.get("efficiency") if isinstance(dims.get("efficiency"), dict) else {}
        for d in (completion, quality):
            if d.get("rating") not in ("pass", "weak", "fail", "n-a"):
                d["rating"] = "n-a"
        return {"completion": completion, "quality": quality, "efficiency": eff}
    except Exception:
        return {
            "completion": {"rating": "n-a", "note": "（agent 审计调用失败，回退 n-a）"},
            "quality": {"rating": "n-a", "note": "（agent 审计调用失败，回退 n-a）"},
            "efficiency": {},
        }


V4_AGG_SYSTEM = """你是 CANNBot Agent 审计聚合器。输入是 session 的 taskQuery + 每个 agent 的三维度评级摘要。
产出：
- sessionSummary：一句话概括本 session 整体完成情况与最突出问题（如：主 agent 因 X 未完成整体交付，N 个子 agent 中 M 个 weak/fail…）。
- crossIssues[]：跨 agent 问题，每项 {type(从 incomplete/out-of-order/redundant/missing-step/session-crash/other 选), severity(high/medium/low), title, detail?, suggestion?}。
- optimizationPriorities[]：按预期收益排序，每项 {priority(从1递增), target(agent:<id> 或 workflow:<step>), action, expectedGain}。
只基于给定摘要，不臆测。输出严格 JSON：{"sessionSummary":"","crossIssues":[],"optimizationPriorities":[]}
"""


def _aggregate_session(provider: AIProviderConfig, task_query: str, summary_list: list) -> dict:
    user = (
        f"taskQuery: {task_query}\n\n"
        f"各 agent 评级摘要（JSON）：\n{json.dumps(summary_list, ensure_ascii=False)}"
    )
    try:
        resp = call_llm(
            provider,
            _load_v4_prompt("audit-v4-agg.md", V4_AGG_SYSTEM),
            [{"role": "user", "content": user}],
        )
        data = _load_json_lenient(resp.content)
        if not isinstance(data, dict):
            return {"sessionSummary": "", "crossIssues": [], "optimizationPriorities": []}
        return {
            "sessionSummary": str(data.get("sessionSummary", ""))[:2000],
            "crossIssues": data.get("crossIssues") if isinstance(data.get("crossIssues"), list) else [],
            "optimizationPriorities": (
                data.get("optimizationPriorities")
                if isinstance(data.get("optimizationPriorities"), list) else []
            ),
        }
    except Exception:
        return {"sessionSummary": "", "crossIssues": [], "optimizationPriorities": []}


def run_analysis_pipeline(
    trajectory_text: str,
    prompt_md: str,
    provider: AIProviderConfig,
    ctx: RunContext,
) -> AnalysisResult:
    """分析入口：agent=v2 循环，claude=v3 CLI，v4=agent-中心三维度审计。"""
    if ctx.mode == "v4" and ctx.agent_io is not None:
        return run_v4_pipeline(ctx.agent_io, prompt_md, ctx, provider)
    if ctx.mode == "claude":
        return run_claude_code_pipeline(trajectory_text, prompt_md, provider, ctx)
    return run_agent_pipeline(trajectory_text, prompt_md, provider, ctx)


def _normalize_usage(usage: dict) -> dict:
    """把 OpenAI 兼容 usage 映射到 Claude Code 格式。"""
    if not usage:
        return {}
    return {
        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
        "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
        "cache_read_input_tokens": (
            usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            if isinstance(usage.get("prompt_tokens_details"), dict) else 0
        ),
        "cache_creation_input_tokens": 0,
    }


def _estimate_cost(usage: dict) -> float:
    """粗略估算成本（dashscope qwen 不返回价格，返 0）。"""
    return 0.0


def analyze_trajectory(
    trajectory_path: str,
    provider: AIProviderConfig,
    prompt_path: str | None = None,
    output_dir: str | None = None,
    mode: str = "agent",
) -> AnalysisResult:
    if prompt_path is None:
        prompt_path = os.path.join(os.getcwd(), "prompts", "session-trajectory-analyse.md")
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "logs")

    if not os.path.exists(trajectory_path):
        raise AnalysisError(f"trajectory file not found: {trajectory_path}")
    if not os.path.exists(prompt_path):
        raise AnalysisError(f"prompt file not found: {prompt_path}")

    trajectory_text = Path(trajectory_path).read_text(encoding="utf-8")
    prompt_md = Path(prompt_path).read_text(encoding="utf-8")
    basename = Path(trajectory_path).stem

    from jsonl_logger import JsonlLogger
    log_path = os.path.join(output_dir, f"smart-agent-{basename}.jsonl")
    logger = JsonlLogger(log_path=log_path, cwd=os.getcwd())

    ctx = RunContext(
        output_dir=output_dir, output_basename=basename,
        logger=logger, mode=mode,
    )
    return run_analysis_pipeline(trajectory_text, prompt_md, provider, ctx)
