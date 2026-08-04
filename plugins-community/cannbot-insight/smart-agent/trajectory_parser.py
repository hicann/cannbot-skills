# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""轨迹 MD 正则提取器（trajectory-parser.ts 的 Python 移植版）。

行号约定：1-based，与 TS 版一致。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, TypedDict


# ---- 正则模式（与 TS TRAJECTORY_PATTERNS 完全一致）----
TRAJECTORY_PATTERNS = {
    "turn": re.compile(r"^## §(\d+) "),
    "skill_call": re.compile(r"^\*Skill: (\S+) \((\w+)\) (✅|❌)"),
    "tool_skill": re.compile(r"^\*\*Tool: skill\*\*"),
    "tool_task": re.compile(r"^\*\*Tool: task\*\*"),
    "skill_content_open": re.compile(r'<skill_content name="([^"]+)">'),
    "skill_content_close": re.compile(r"</skill_content>"),
    "tool_boundary": re.compile(r"^\*\*Tool: "),
    "task_result_open": re.compile(r"<task_result>"),
    "task_result_close": re.compile(r"</task_result>"),
    "pass": re.compile(r"^(PASS)\s*$"),
    "fail": re.compile(r"^(FAILED|FAIL)\s*$"),
    "skill_err": re.compile(r"^\*Skill:.*❌"),
    "tool_err": re.compile(r"^\*Error:"),
    "duration": re.compile(r"\*\*Duration:\*\*\s*(\S+)"),
    "tokens": re.compile(r"\*\*Tokens:\*\*\s*([^|]+)"),
    "stats_row": re.compile(r"\|\s*(\w+)\s*\|\s*[^|]+\|\s*[^|]+\|\s*([^|]+?)\s*\|"),
    "section_h": re.compile(r"^### \*\*§(\d+\.\d+)\*\*"),
}


@dataclass
class SkeletonEntry:
    turn: int
    skill: str
    type: str
    status: Literal["ok", "fail"]
    line: int


@dataclass
class SkeletonResult:
    skeleton: list[SkeletonEntry] = field(default_factory=list)
    occurrences: dict[str, int] = field(default_factory=dict)
    turn_count: int = 0


def extract_skeleton(text: str) -> SkeletonResult:
    lines = text.split("\n")
    skeleton: list[SkeletonEntry] = []
    occurrences: dict[str, int] = {}
    turn_seen: set[int] = set()
    current_turn = 0

    for i, line in enumerate(lines):
        tm = TRAJECTORY_PATTERNS["turn"].search(line)
        if tm:
            current_turn = int(tm.group(1))
            turn_seen.add(current_turn)
            continue
        sm = TRAJECTORY_PATTERNS["skill_call"].search(line)
        if sm:
            skill, stype, mark = sm.group(1), sm.group(2), sm.group(3)
            skeleton.append(SkeletonEntry(
                turn=current_turn,
                skill=skill,
                type=stype,
                status="ok" if mark == "✅" else "fail",
                line=i + 1,
            ))
            occurrences[skill] = occurrences.get(skill, 0) + 1

    return SkeletonResult(skeleton=skeleton, occurrences=occurrences, turn_count=len(turn_seen))


@dataclass
class SkillContentResult:
    skill_md: str = ""
    skill_name: str = ""


def _find_close_tag(lines: list[str], from_idx: int) -> int:
    for i in range(from_idx, len(lines)):
        if TRAJECTORY_PATTERNS["skill_content_close"].search(lines[i]):
            return i
    return -1


def _find_skill_content_at(lines: list[str], start_idx: int) -> SkillContentResult | None:
    for j in range(start_idx, min(len(lines), start_idx + 500)):
        open_m = TRAJECTORY_PATTERNS["skill_content_open"].search(lines[j])
        if open_m:
            skill_name = open_m.group(1)
            close_idx = _find_close_tag(lines, j + 1)
            if close_idx == -1:
                break
            return SkillContentResult(
                skill_md="\n".join(lines[j:close_idx + 1]),
                skill_name=skill_name,
            )
    return None


def extract_skill_content(text: str) -> SkillContentResult:
    lines = text.split("\n")
    invoke_skill_lines: list[int] = []
    for i, line in enumerate(lines):
        sm = TRAJECTORY_PATTERNS["skill_call"].search(line)
        if sm and sm.group(2) == "invoke":
            invoke_skill_lines.append(i)

    for start_idx in invoke_skill_lines:
        result = _find_skill_content_at(lines, start_idx)
        if result is not None:
            return result
    return SkillContentResult()


class StatsResult(TypedDict):
    duration: str
    tokens: str
    turns: str
    subagents: str
    cost: str


def parse_stats(text: str) -> StatsResult:
    duration_m = TRAJECTORY_PATTERNS["duration"].search(text)
    tokens_m = TRAJECTORY_PATTERNS["tokens"].search(text)

    turns = ""
    subagents = ""
    cost = ""

    for line in text.split("\n"):
        rm = TRAJECTORY_PATTERNS["stats_row"].search(line)
        if rm:
            metric, total = rm.group(1), rm.group(2)
            if metric == "Turns":
                turns = total.strip()
            elif metric == "Subagents":
                subagents = total.strip()
            elif metric == "Cost":
                cost = total.strip()

    return StatsResult(
        duration=duration_m.group(1) if duration_m else "",
        tokens=(tokens_m.group(1).strip() if tokens_m else ""),
        turns=turns,
        subagents=subagents,
        cost=cost,
    )


@dataclass
class GateEntry:
    turn: int
    result: Literal["PASS", "FAIL"]
    line: int
    snippet: str


def extract_gates(text: str) -> list[GateEntry]:
    lines = text.split("\n")
    gates: list[GateEntry] = []
    current_turn = 0
    in_details = 0

    for i, line in enumerate(lines):
        if re.search(r"^\s*<details", line):
            in_details += 1
        if re.search(r"^\s*</details>", line):
            in_details = max(0, in_details - 1)

        tm = TRAJECTORY_PATTERNS["turn"].search(line)
        if tm:
            current_turn = int(tm.group(1))
            continue

        if in_details > 0:
            continue

        if TRAJECTORY_PATTERNS["pass"].search(line):
            gates.append(GateEntry(turn=current_turn, result="PASS", line=i + 1, snippet=line))
        elif TRAJECTORY_PATTERNS["fail"].search(line):
            gates.append(GateEntry(turn=current_turn, result="FAIL", line=i + 1, snippet=line))

    return gates


@dataclass
class ErrorEntry:
    turn: int
    line_range: tuple[int, int]
    snippet: str


def extract_errors(text: str, context_lines: int = 15) -> list[ErrorEntry]:
    lines = text.split("\n")
    errors: list[ErrorEntry] = []
    current_turn = 0

    for i, line in enumerate(lines):
        tm = TRAJECTORY_PATTERNS["turn"].search(line)
        if tm:
            current_turn = int(tm.group(1))
            continue

        is_skill_err = bool(TRAJECTORY_PATTERNS["skill_err"].search(line))
        is_tool_err = bool(TRAJECTORY_PATTERNS["tool_err"].search(line))
        if not is_skill_err and not is_tool_err:
            continue

        frm = max(0, i - context_lines)
        to = min(len(lines) - 1, i + context_lines)
        errors.append(ErrorEntry(
            turn=current_turn,
            line_range=(frm + 1, to + 1),
            snippet="\n".join(lines[frm:to + 1]),
        ))

    return errors


# ---- 关键片段提取（dispatch 子代理 task prompt + task_result）----


@dataclass
class KeySection:
    turn: int
    skill: str
    line: int
    task_prompt: str
    task_result: str


def _cap(text: str, limit: int = 2000) -> str:
    return text[:limit] if len(text) > limit else text


def extract_key_sections(text: str, skeleton: SkeletonResult) -> list[KeySection]:
    """对每个 dispatch 骨架项，提取 **Tool: task** 块（task prompt）和
    <task_result>...</task_result> 块（若存在）。

    invoke 类型跳过（skill_content 已由 extract_skill_content 单独提取）。
    """
    lines = text.split("\n")
    n = len(lines)
    sections: list[KeySection] = []

    for entry in skeleton.skeleton:
        if entry.type != "dispatch":
            continue
        start_idx = entry.line - 1  # 0-based
        if start_idx < 0 or start_idx >= n:
            continue

        # 向前扫描 **Tool: task** 标记（可能在 entry.line 当行或之后几行）
        task_line_idx = -1
        for i in range(start_idx, min(n, start_idx + 50)):
            if TRAJECTORY_PATTERNS["tool_task"].search(lines[i]):
                task_line_idx = i
                break
        if task_line_idx == -1:
            continue

        # task_prompt：从 **Tool: task** 行到下一个 **Tool: 边界或 --- 分隔符
        prompt_end = task_line_idx + 1
        for i in range(task_line_idx + 1, min(n, task_line_idx + 200)):
            line = lines[i]
            if TRAJECTORY_PATTERNS["tool_boundary"].search(line):
                prompt_end = i
                break
            if line.strip() == "---":
                prompt_end = i
                break
        else:
            prompt_end = min(n, task_line_idx + 200)
        task_prompt = "\n".join(lines[task_line_idx:prompt_end])

        # task_result：在 task_prompt 结束后 ~200 行内找 <task_result>...</task_result>
        task_result = ""
        search_start = prompt_end
        search_end = min(n, search_start + 200)
        tr_open_idx = -1
        tr_close_idx = -1
        for i in range(search_start, search_end):
            if tr_open_idx == -1 and TRAJECTORY_PATTERNS["task_result_open"].search(lines[i]):
                tr_open_idx = i
            if tr_open_idx != -1 and TRAJECTORY_PATTERNS["task_result_close"].search(lines[i]):
                tr_close_idx = i
                break
        if tr_open_idx != -1 and tr_close_idx != -1 and tr_close_idx >= tr_open_idx:
            task_result = "\n".join(lines[tr_open_idx:tr_close_idx + 1])

        sections.append(KeySection(
            turn=entry.turn,
            skill=entry.skill,
            line=task_line_idx + 1,
            task_prompt=_cap(task_prompt),
            task_result=_cap(task_result),
        ))

    return sections


# ---- read request（LLM 请求补上下文）----
ReadRequestLines = tuple[int, int]  # (from, to)


class ReadRequestSection(TypedDict):
    section: str


def read_section(text: str, req: dict) -> str:
    """req 形如 {"lines": [from, to]} 或 {"section": "§N.M"}。"""
    lines = text.split("\n")

    if "lines" in req:
        frm, to = req["lines"]
        start = max(0, frm - 1)
        end = min(len(lines), to)
        return "\n".join(lines[start:end])

    if "section" in req:
        target = req["section"].lstrip("§")
        start_idx = -1
        for i, line in enumerate(lines):
            m = TRAJECTORY_PATTERNS["section_h"].search(line)
            if m and m.group(1) == target:
                start_idx = i
                break
        if start_idx == -1:
            return ""
        end_idx = len(lines)
        for i in range(start_idx + 1, len(lines)):
            m = TRAJECTORY_PATTERNS["section_h"].search(lines[i])
            if m:
                end_idx = i
                break
        return "\n".join(lines[start_idx:end_idx])

    return ""


# ---- Turn 指标提取（耗时 + token）----

@dataclass
class TurnMetrics:
    turn: int
    duration_sec: float
    tokens_kt: float
    model: str = ""


@dataclass
class SessionMetrics:
    """子代理 session（§N.M）的聚合指标"""
    session: str          # "§4.1"
    root_turn: int        # 4
    sub_index: int        # 1
    duration_sec: float
    tokens_kt: float
    turn_count: int
    skill: str = ""


_TURN_HEADER_RE = re.compile(
    r'^## §(\d+) (?:User|Assistant)(?:.*?·\s*([\d.]+)(s|min)\s*·.*?(\d+(?:\.\d+)?)Kt)?',
    re.MULTILINE,
)
_SUB_TURN_HEADER_RE = re.compile(
    r'^#### §(\d+)\.(\d+)\.(\d+) (?:User|Assistant)(?:.*?·\s*([\d.]+)(s|min)\s*·.*?(\d+(?:\.\d+)?)Kt)?',
    re.MULTILINE,
)
_SESSION_HEADER_RE = re.compile(
    r'^### \*\*§(\d+)\.(\d+)\*\* (\S+).*?(\d+) turns.*?([\d.]+)Kt.*?([\d.]+)(s|min)',
    re.MULTILINE,
)


def _parse_turn_headers(text: str) -> dict[int, dict]:
    turns: dict[int, dict] = {}
    for m in _TURN_HEADER_RE.finditer(text):
        tn = int(m.group(1))
        dur = float(m.group(2)) * (60 if m.group(3) == "min" else 1) if m.group(2) else 0
        tokens = float(m.group(4)) if m.group(4) else 0
        turns[tn] = {"duration_sec": dur, "tokens_kt": tokens}
    return turns


def _parse_session_headers(text: str) -> dict[str, dict]:
    sessions: dict[str, dict] = {}
    for m in _SESSION_HEADER_RE.finditer(text):
        rt, si = int(m.group(1)), int(m.group(2))
        skill = m.group(3)
        tc = int(m.group(4))
        tokens = float(m.group(5))
        dur = float(m.group(6)) * (60 if m.group(7) == "min" else 1)
        key = f"§{rt}.{si}"
        sessions[key] = {
            "duration_sec": dur,
            "tokens_kt": tokens,
            "turn_count": tc,
            "skill": skill,
            "root_turn": rt,
            "sub_index": si,
        }
    return sessions


def _aggregate_sub_turns(text: str) -> dict[str, dict]:
    sub_turns: dict[tuple[int, int], list] = {}
    for m in _SUB_TURN_HEADER_RE.finditer(text):
        rt, si, st = int(m.group(1)), int(m.group(2)), int(m.group(3))
        dur = float(m.group(4)) * (60 if m.group(5) == "min" else 1) if m.group(4) else 0
        tokens = float(m.group(6)) if m.group(6) else 0
        sub_turns.setdefault((rt, si), []).append({"dur": dur, "tokens": tokens})
    sessions: dict[str, dict] = {}
    for (rt, si), items in sub_turns.items():
        key = f"§{rt}.{si}"
        sessions[key] = {
            "duration_sec": sum(i["dur"] for i in items),
            "tokens_kt": sum(i["tokens"] for i in items),
            "turn_count": len(items),
            "skill": "",
            "root_turn": rt,
            "sub_index": si,
        }
    return sessions


def _build_metrics_result(turns: dict, sessions: dict) -> dict:
    total_dur = sum(t["duration_sec"] for t in turns.values()) + sum(s["duration_sec"] for s in sessions.values())
    total_tokens = sum(t["tokens_kt"] for t in turns.values()) + sum(s["tokens_kt"] for s in sessions.values())
    return {
        "turns": turns,
        "sessions": sessions,
        "total": {
            "duration_sec": total_dur,
            "tokens_kt": total_tokens,
            "main_turns": len(turns),
            "sub_turns": sum(s["turn_count"] for s in sessions.values()),
            "session_count": len(sessions),
        },
    }


def extract_turn_metrics(text: str) -> dict:
    """提取每个 turn 和子代理 session 的耗时与 token 数据。

    返回:
      {
        "turns": {turn_num: {duration_sec, tokens_kt}},          # 主 Agent 每 turn
        "sessions": {"§N.M": {duration_sec, tokens_kt, turn_count, skill, root_turn}},  # 子代理 session
        "total": {duration_sec, tokens_kt, main_turns, sub_turns, session_count}
      }
    """
    turns = _parse_turn_headers(text)
    sessions = _parse_session_headers(text)
    if not sessions:
        sessions = _aggregate_sub_turns(text)
    return _build_metrics_result(turns, sessions)


def inject_flow_metrics(analysis: dict, trajectory_text: str) -> dict:
    """给 analysis JSON 的 flow 节点注入耗时和 token 数据。

    每个 flow 节点有 turn 字段——用它查找：
    - dispatch/gate/terminal 节点 → 查子代理 session（§turn.*）的汇总
    - invoke 节点 → 查主 Agent turn 的耗时
    """
    metrics = extract_turn_metrics(trajectory_text)
    turns = metrics["turns"]
    sessions = metrics["sessions"]

    flow = analysis.get("flow", [])
    for node in flow:
        turn = node.get("turn")
        if turn is None:
            continue
        turn = int(turn)

        # 找以这个 turn 为根的子代理 session
        session_keys = [k for k, v in sessions.items() if v["root_turn"] == turn]
        if session_keys:
            # 汇总该 turn 下所有子代理 session 的指标
            dur = sum(sessions[k]["duration_sec"] for k in session_keys)
            tokens = sum(sessions[k]["tokens_kt"] for k in session_keys)
            tc = sum(sessions[k]["turn_count"] for k in session_keys)
            node["durationSec"] = round(dur, 1)
            node["tokensKt"] = round(tokens, 1)
            node["turnCount"] = tc
            if len(session_keys) == 1:
                node["subSessions"] = session_keys
            else:
                node["subSessions"] = session_keys
        elif turn in turns:
            # 主 Agent turn（如 invoke 节点）
            node["durationSec"] = round(turns[turn]["duration_sec"], 1)
            node["tokensKt"] = round(turns[turn]["tokens_kt"], 1)
            node["turnCount"] = 1

    # 注入 perfAnalysis 的总览数据
    pa = analysis.get("perfAnalysis", {})
    if pa:
        pa["totalDurationSec"] = round(metrics["total"]["duration_sec"], 0)
        pa["totalTokensKt"] = round(metrics["total"]["tokens_kt"], 0)
        pa["mainTurns"] = metrics["total"]["main_turns"]
        pa["subagentSessions"] = metrics["total"]["session_count"]
        pa["subagentTurns"] = metrics["total"]["sub_turns"]
        pa["parallelSessions"] = sum(1 for n in flow if n.get("parallel"))
        pa["serialRatio"] = round(1 - pa["parallelSessions"] / max(len(flow), 1), 2)
        analysis["perfAnalysis"] = pa

    return analysis
