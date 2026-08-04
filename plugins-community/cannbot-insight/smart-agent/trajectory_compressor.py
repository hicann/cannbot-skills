# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""轨迹压缩器：per-turn LLM 压缩 → 新 MD 文件。

流程：split_turns → classify_turn → compress_turns → assemble_md
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from trajectory_analyzer import AIProviderConfig, RunContext, call_llm_text

# ---- 正则 ----
_MAIN_TURN_RE = re.compile(r"^## §(\d+) ")
_SUB_TURN_RE = re.compile(r"^#### §(\d+\.\d+\.\d+) ")
_STATS_RE = re.compile(r"^## Stats\b")
_SKILL_CONTENT_RE = re.compile(r'<skill_content[^>]*>.*?</skill_content>', re.DOTALL)
_WORKFLOW_MARKER_RE = re.compile(
    r'\*Skill: \S+ (invoke|dispatch)|^\*\*Tool: (skill|todowrite|task)\*\*',
    re.MULTILINE,
)
_TOOL_BOUNDARY_RE = re.compile(r'^\*\*Tool: ', re.MULTILINE)
# 受保护工具块（skill/task/todowrite 的 Input/Output 原样保留，不压缩）
_PROTECTED_TOOL_RE = re.compile(
    r'(^\*\*Tool: (?:skill|task|todowrite)\*\*\n.*?)(?=^\*\*Tool: |\Z)',
    re.MULTILINE | re.DOTALL,
)

# 结构性行（不压缩，从 turn 内容尾部剥离后原样回贴）
_STRUCTURAL_LINE_PATTERNS = [
    re.compile(r'^\s*</details>\s*$'),
    re.compile(r'^\s*<details'),
    re.compile(r'^\s*<summary>.*</summary>\s*$'),
    re.compile(r'^\s*### \*\*§'),
    re.compile(r'^---+\s*$'),
]

SMALL_TURN_THRESHOLD = 100      # <100 行 = 小 turn
BATCH_SIZE = 12                 # 小 turn 每批最多 12 个
MAX_WORKERS = 15                # 最多 15 个并行 LLM call
CHUNK_MAX_LINES = 1500          # 大 turn 切片每 chunk 最多行数


@dataclass
class TurnBlock:
    header: str
    content: str
    line_start: int
    line_end: int
    is_workflow: bool = False
    skill_contents: list[str] = field(default_factory=list)
    compressed_content: str = ""
    was_compressed: bool = False


def _is_structural_line(line: str) -> bool:
    return any(p.search(line) for p in _STRUCTURAL_LINE_PATTERNS)


def _split_trailing_structure(content: str) -> tuple[str, str]:
    """从内容尾部剥离结构性行（</details>、---、### **§N.M** 等），返回 (body, trailer)。"""
    lines = content.split("\n")
    trailer_start = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if _is_structural_line(lines[i]):
            trailer_start = i
        else:
            break
    if trailer_start == len(lines):
        return content, ""
    body = "\n".join(lines[:trailer_start])
    trailer = "\n".join(lines[trailer_start:])
    return body, trailer


def split_turns(text: str) -> tuple[str, list[TurnBlock], str]:
    """按 ## §N 和 #### §N.M.X 分割，返回 (preamble, turns, postamble)。

    preamble = 第一个 ## § 之前的内容（含标题块、Session 元信息）
    postamble = ## Stats 段（文件末尾的统计表）
    """
    lines = text.split("\n")
    preamble_lines: list[str] = []
    turns: list[TurnBlock] = []
    postamble_lines: list[str] = []

    current_turn: TurnBlock | None = None
    in_postamble = False

    for i, line in enumerate(lines):
        if _STATS_RE.match(line):
            # Stats 段开始：结束当前 turn，进入 postamble
            if current_turn is not None:
                current_turn.content = current_turn.content.rstrip("\n")
                turns.append(current_turn)
                current_turn = None
            in_postamble = True
            postamble_lines.append(line)
            continue

        if in_postamble:
            postamble_lines.append(line)
            continue

        if _MAIN_TURN_RE.match(line) or _SUB_TURN_RE.match(line):
            if current_turn is not None:
                current_turn.content = current_turn.content.rstrip("\n")
                turns.append(current_turn)
            current_turn = TurnBlock(
                header=line,
                content="",
                line_start=i + 1,
                line_end=i + 1,
            )
            continue

        if current_turn is not None:
            current_turn.content += line + "\n"
            current_turn.line_end = i + 1
        else:
            preamble_lines.append(line)

    if current_turn is not None:
        current_turn.content = current_turn.content.rstrip("\n")
        turns.append(current_turn)

    return "\n".join(preamble_lines), turns, "\n".join(postamble_lines)


def classify_turn(turn: TurnBlock) -> TurnBlock:
    """标记 is_workflow，提取 skill_content 块。"""
    if _WORKFLOW_MARKER_RE.search(turn.content):
        turn.is_workflow = True

    skill_contents = _SKILL_CONTENT_RE.findall(turn.content)
    if skill_contents:
        turn.skill_contents = skill_contents

    return turn


def _strip_protected(content: str) -> tuple[str, list[str]]:
    """提取受保护块（skill_content + skill/task/todowrite 工具块），替换为占位符。

    返回 (stripped_content, blocks)：blocks[i] 对应占位符 <!-- keep_i -->。
    先保护 skill_content，再保护 skill/task/todowrite 工具块——后者的 Output
    常含 skill_content 占位符，故高 idx 内嵌低 idx，回填需逆序。
    """
    blocks: list[str] = []

    def _replace(m: re.Match) -> str:
        idx = len(blocks)
        blocks.append(m.group(0))
        return f"<!-- keep_{idx} -->"

    stripped = _SKILL_CONTENT_RE.sub(_replace, content)
    stripped = _PROTECTED_TOOL_RE.sub(_replace, stripped)
    return stripped, blocks


def _restore_protected(content: str, blocks: list[str]) -> str:
    # 逆序回填：高 idx 的工具块可能内嵌低 idx 的 skill_content 占位符
    for i in range(len(blocks) - 1, -1, -1):
        content = content.replace(f"<!-- keep_{i} -->", blocks[i])
    return content


def _split_into_chunks(content: str, max_lines: int = CHUNK_MAX_LINES) -> list[str]:
    """按 **Tool:** 边界切分大 turn 内容为多个 chunk。"""
    lines = content.split("\n")
    if len(lines) <= max_lines:
        return [content]

    chunks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if _TOOL_BOUNDARY_RE.match(line) and len(current) >= max_lines:
            chunks.append("\n".join(current))
            current = []
        current.append(line)

    if current:
        chunks.append("\n".join(current))

    return chunks


def _compress_single_turn(turn: TurnBlock, provider: AIProviderConfig, prompt_md: str) -> str:
    """压缩单个非 workflow turn，返回压缩后内容（含 skill_content 回填 + trailer 回贴）。"""
    body, trailer = _split_trailing_structure(turn.content)
    stripped, blocks = _strip_protected(body)

    if not stripped.strip():
        # 无可压缩内容
        return turn.content

    chunks = _split_into_chunks(stripped)
    compressed_chunks: list[str] = []

    for chunk in chunks:
        user_text = (
            "请压缩以下 turn 内容（保持 MD 结构，"
            "所有 <!-- keep_N --> 占位符原样保留）：\n\n"
            + chunk
        )
        result = call_llm_text(provider, prompt_md, user_text)
        compressed_chunks.append(result.strip())

    compressed_body = "\n\n".join(compressed_chunks)
    compressed_body = _restore_protected(compressed_body, blocks)

    parts = [compressed_body]
    if trailer:
        parts.append(trailer)
    return "\n".join(parts)


def _compress_batch(
    turns: list[TurnBlock],
    provider: AIProviderConfig,
    prompt_md: str,
) -> list[str]:
    """批量压缩多个小 turn，一次 LLM call 完成。返回与 turns 等长的压缩结果列表。"""
    if not turns:
        return []

    parts: list[str] = []
    for idx, turn in enumerate(turns):
        body, trailer = _split_trailing_structure(turn.content)
        stripped, blocks = _strip_protected(body)
        parts.append(
            f'<compress_turn index="{idx}" header="{turn.header[:120]}">\n'
            f"{stripped}\n"
            f"</compress_turn>"
        )
        # 保存提取的 blocks 供回填
        turn.batch_protected_blocks = blocks  # type: ignore[attr-defined]
        turn.batch_trailer = trailer  # type: ignore[attr-defined]

    user_text = (
        "请压缩以下多个 turn 内容。每个 <compress_turn> 标签内是一个 turn，"
        "输出时保留 <compress_turn index=\"N\"> 和 </compress_turn> 标签，"
        "所有 <!-- keep_N --> 占位符原样保留。各 turn 之间用空行分隔。\n\n"
        + "\n\n".join(parts)
    )

    result = call_llm_text(provider, prompt_md, user_text)

    # 解析 LLM 返回的 <compress_turn> 块
    parsed: dict[int, str] = {}
    pattern = re.compile(
        r'<compress_turn index="(\d+)"[^>]*>\n?(.*?)</compress_turn>',
        re.DOTALL,
    )
    for m in pattern.finditer(result):
        idx = int(m.group(1))
        parsed[idx] = m.group(2).strip()

    outputs: list[str] = []
    for idx, turn in enumerate(turns):
        compressed = parsed.get(idx, turn.content)  # fallback：原文
        blocks = turn.batch_protected_blocks  # type: ignore[attr-defined]
        trailer = turn.batch_trailer  # type: ignore[attr-defined]
        compressed = _restore_protected(compressed, blocks)
        parts_out = [compressed]
        if trailer:
            parts_out.append(trailer)
        outputs.append("\n".join(parts_out))

    return outputs


def _classify_compress(turns: list[TurnBlock], on_progress, logger) -> None:
    workflow_count = sum(1 for t in turns if t.is_workflow)
    msg = f"共 {len(turns)} 个 turn（其中 {workflow_count} 个含 workflow marker，受保护块原样保留）"
    if on_progress:
        on_progress({"stage": "classify", "msg": msg})
    if logger:
        logger.log_system("progress", {"stage": "classify", "msg": msg})


def _split_turns_by_size(turns: list[TurnBlock]) -> tuple[list, list]:
    small_turns: list[tuple[int, TurnBlock]] = []
    single_turns: list[tuple[int, TurnBlock]] = []
    for i, t in enumerate(turns):
        line_count = t.content.count("\n") + 1
        if line_count < SMALL_TURN_THRESHOLD:
            small_turns.append((i, t))
        else:
            single_turns.append((i, t))
    return small_turns, single_turns


def _do_single(item: tuple[int, TurnBlock], provider: AIProviderConfig,
               prompt_md: str, on_progress) -> tuple[int, str]:
    idx, turn = item
    line_count = turn.content.count("\n") + 1
    if on_progress:
        on_progress({
            "stage": "compress",
            "turn": idx,
            "msg": f"压缩 {turn.header[:60].strip()} ({line_count} lines)…",
        })
    compressed = _compress_single_turn(turn, provider, prompt_md)
    return idx, compressed


def _do_batch(batch: list[tuple[int, TurnBlock]], provider: AIProviderConfig,
              prompt_md: str, on_progress) -> list[tuple[int, str]]:
    turn_refs = [t for _, t in batch]
    headers_preview = ", ".join(t.header[:30].strip() for t in turn_refs[:3])
    if on_progress:
        on_progress({
            "stage": "compress-batch",
            "count": len(batch),
            "msg": f"批量压缩 {len(batch)} 个小 turn（{headers_preview}…）",
        })
    results = _compress_batch(turn_refs, provider, prompt_md)
    return [(batch[k][0], results[k]) for k in range(len(batch))]


def _merge_result(ret, results: dict[int, str]) -> None:
    if isinstance(ret, list):
        for idx, compressed in ret:
            results[idx] = compressed
    else:
        idx, compressed = ret
        results[idx] = compressed


def _log_compress_error(e: Exception, kind: str, on_progress, logger) -> None:
    err_msg = f"压缩失败：{type(e).__name__}: {e}"
    if on_progress:
        on_progress({"stage": "error", "msg": err_msg})
    if logger:
        logger.log_system("error", {"msg": err_msg})


def _run_compress_tasks(all_tasks: list, provider: AIProviderConfig, prompt_md: str,
                        on_progress, logger) -> dict[int, str]:
    results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {}
        for kind, payload in all_tasks:
            if kind == "single":
                fut = executor.submit(_do_single, payload, provider, prompt_md, on_progress)
            else:
                fut = executor.submit(_do_batch, payload, provider, prompt_md, on_progress)
            future_map[fut] = kind

        for fut in as_completed(future_map):
            try:
                _merge_result(fut.result(), results)
            except Exception as e:
                _log_compress_error(e, future_map[fut], on_progress, logger)
    return results


def _apply_compressed(turns: list[TurnBlock], results: dict[int, str]) -> list[TurnBlock]:
    for i, t in enumerate(turns):
        if t.is_workflow:
            continue
        if i in results:
            t.compressed_content = results[i]
            t.was_compressed = True
    return turns


def compress_turns(
    turns: list[TurnBlock],
    provider: AIProviderConfig,
    prompt_md: str,
    on_progress=None,
    logger=None,
) -> list[TurnBlock]:
    """对所有 turn 执行 LLM 压缩（受保护块原样保留，普通块正常压缩）。"""
    _classify_compress(turns, on_progress, logger)

    small_turns, single_turns = _split_turns_by_size(turns)
    batches = [small_turns[j:j + BATCH_SIZE] for j in range(0, len(small_turns), BATCH_SIZE)]
    all_tasks = [("single", item) for item in single_turns] + [("batch", batch) for batch in batches]

    results = _run_compress_tasks(all_tasks, provider, prompt_md, on_progress, logger)
    return _apply_compressed(turns, results)


def assemble_md(
    preamble: str,
    turns: list[TurnBlock],
    postamble: str,
) -> str:
    """重新组装为完整 MD 文本。"""
    parts: list[str] = []

    if preamble.strip():
        parts.append(preamble.rstrip("\n"))

    for turn in turns:
        if turn.was_compressed and turn.compressed_content:
            parts.append(turn.header + "\n" + turn.compressed_content.strip("\n"))
        else:
            parts.append(turn.header + "\n" + turn.content.strip("\n"))

    if postamble.strip():
        parts.append(postamble.rstrip("\n"))

    return "\n\n".join(parts) + "\n"


# 确定性压缩用的边界正则（下一个 Tool / turn 标题 / 结构标签 / Stats）
_BOUNDARY_LINE_RE = re.compile(
    r'^(\*\*Tool: |## §|#### §|## Stats\b|### \*\*§|\s*</details>|\s*<details|\s*<summary>|---+\s*$)'
)
_PROTECTED_TOOL_NAMES = {"skill", "task", "todowrite"}


def _compress_tool_block(lines: list[str], i: int, out: list[str]) -> int:
    """保留 **Tool: X** + Input，截断 **Output:** 段到下个边界。返回下一个待处理行号。"""
    out.append(lines[i])  # **Tool: X**
    i += 1
    n = len(lines)
    while i < n:
        l = lines[i]
        if l.strip() == "**Output:**":
            i += 1
            # 跳过 Output 内容直到边界
            while i < n and not _BOUNDARY_LINE_RE.match(lines[i]):
                i += 1
            break
        if _BOUNDARY_LINE_RE.match(l):
            break
        out.append(l)
        i += 1
    return i


def deterministic_compress(text: str) -> str:
    """确定性压缩（秒级，不依赖 LLM）：
    截断 bash/read/edit/write/webfetch/grep/glob 工具块的 Output 段（保留 Input）。
    skill/task/todowrite 工具块、skill_content、_Thinking: 块、turn 标题、门控行、
    结构标签原样保留——thinking 含 agent 决策推理，claude code 分析时需要，
    不机械截断（由审计 prompt 指导 claude code 按需精读/跳过）。
    """
    lines = text.split("\n")
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        m = re.match(r'^\*\*Tool: (\w+)\*\*\s*$', line)
        if not m or m.group(1).lower() in _PROTECTED_TOOL_NAMES:
            out.append(line)
            i += 1
            continue
        # 非保护工具块：保留 **Tool: X** + Input，截断 **Output:** 段到下个边界
        i = _compress_tool_block(lines, i, out)
    return "\n".join(out)


def compress_trajectory(
    text: str,
    provider: AIProviderConfig,
    prompt_path: str,
    ctx: RunContext,
) -> str:
    """入口：返回压缩后 MD 文件路径。"""
    prompt_md = Path(prompt_path).read_text(encoding="utf-8")

    if ctx.on_progress:
        ctx.on_progress({"stage": "start", "msg": "开始压缩轨迹…"})
    if ctx.logger:
        ctx.logger.log_system("progress", {"stage": "start", "msg": "开始压缩轨迹…"})

    preamble, turns, postamble = split_turns(text)
    if ctx.on_progress:
        ctx.on_progress({"stage": "split", "msg": f"分割为 {len(turns)} 个 turn"})
    if ctx.logger:
        ctx.logger.log_system("progress", {"stage": "split", "msg": f"分割为 {len(turns)} 个 turn"})

    for turn in turns:
        classify_turn(turn)

    turns = compress_turns(turns, provider, prompt_md, ctx.on_progress, ctx.logger)

    md = assemble_md(preamble, turns, postamble)

    out_dir = Path(ctx.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ctx.output_basename}-compressed.md"
    out_path.write_text(md, encoding="utf-8")

    if ctx.on_progress:
        ctx.on_progress({"stage": "result", "outputPath": str(out_path)})
    if ctx.logger:
        ctx.logger.log_system("progress", {"stage": "result", "outputPath": str(out_path)})

    return str(out_path)
