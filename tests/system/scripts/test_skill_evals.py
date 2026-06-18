# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import logging
import os
import glob as glob_module
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pytest

from conftest import (
    get_skill_path, get_skills_with_evals, load_evals_md, REPO_ROOT,
    get_opencode_text, FRAMEWORK_DIR, SANDBOX_DIR,
    parse_dimension_scores, parse_review_md,
    validate_dimension_scores, DEFAULT_DIMENSION_THRESHOLDS,
    DIMENSION_ORDER, DIMENSION_MAX_SCORES,
    create_opencode_runner, _platform_matches,
)
from opencode_runner import OpencodeRunner
from sandbox_manager import SandboxManager

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

# file_based 模式下注入到用户 prompt 的执行要求
FILE_BASED_HINT = """

## 输出要求
1. 完成任务后，列出所有创建/修改的文件路径清单（如 src/main.cpp）
2. 简要说明每个文件的用途
3. 不要输出完整的文件内容——评测系统会直接读取生成的文件
"""

MAX_PROMPT_LENGTH = 10000


def _validate_prompt(prompt: str, eval_id: str) -> None:
    """校验 prompt 安全性，拒绝可疑输入。

    针对不可信来源的 eval prompt 执行基本安全检查。
    """
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(
            f"Eval {eval_id}: prompt too long ({len(prompt)} chars, max {MAX_PROMPT_LENGTH})"
        )
    if prompt.lstrip() != prompt:
        raise ValueError(
            f"Eval {eval_id}: prompt has leading whitespace"
        )
    # 拒绝包含不可打印控制字符的 prompt（允许换行、制表符）
    for i, ch in enumerate(prompt):
        if ord(ch) < 0x20 and ch not in ("\n", "\r", "\t"):
            raise ValueError(
                f"Eval {eval_id}: prompt contains control char U+{ord(ch):04X} at position {i}"
            )
    # 拒绝以 - 开头的 prompt，防止被解析为 opencode CLI 参数
    stripped = prompt.lstrip()
    if stripped.startswith("-"):
        raise ValueError(
            f"Eval {eval_id}: prompt starts with '-', potential CLI injection"
        )


def collect_generated_files(sandbox_path: Path, original_skill_dir: Optional[Path] = None) -> List[str]:
    """收集沙箱中新增的生成文件列表（相对路径）

    排除 logs/ 和 .opencode/ 目录。软链接模式下自动排除源 skill 目录中已存在的文件。
    """
    files = []
    exclude_dirs = {"logs", ".opencode"}
    for entry in sandbox_path.rglob("*"):
        if not entry.is_file():
            continue
        if any(d in entry.parts for d in exclude_dirs):
            continue

        # 软链接模式：排除源 skill 目录中已存在的文件（被 rglob 通过软链接追踪到）
        if original_skill_dir is not None:
            try:
                entry.resolve().relative_to(original_skill_dir.resolve())
                continue
            except ValueError:
                pass

        rel = str(entry.relative_to(sandbox_path))
        files.append(rel)
    return sorted(files)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


@dataclass
class ValidationContext:
    """封装输出验证所需的上下文参数"""
    opencode_runner: OpencodeRunner
    session_name: str
    full_output: str
    original_prompt: str = ""
    expected_output: Optional[str] = None
    expectations: Optional[List[Dict[str, Any]]] = None
    skill_dir: Optional[Path] = None
    eval_id: Optional[str] = None
    ai_text: str = ""
    truncate_len: int = 30000
    # file_based 模式字段
    eval_mode: str = "text"
    sandbox_path: Optional[Path] = None
    generated_files: List[str] = field(default_factory=list)
    # 正向看护字段
    skill_name: str = ""
    # 维度阈值覆盖（可选，None 则使用 conftest 中的默认值）
    dim_thresholds: Optional[Dict[str, int]] = None


@dataclass
class ExpectationContext:
    """封装期望验证所需的参数"""
    exp: Dict[str, Any]
    full_output: str
    ai_text: str
    skill_dir: Optional[Path] = None
    sandbox_path: Optional[Path] = None
    eval_id: Optional[str] = None
    truncate_len: int = 30000
    skill_name: str = ""
    session_export_path: Optional[Path] = None


def _parse_json_line(line: str) -> Optional[Dict[str, Any]]:
    """安全解析单行 JSON"""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def extract_ai_text(full_output: str) -> str:
    """从 full_output 的 JSON 行中提取 AI 文本回复，便于问题定位"""
    texts = []
    for line in full_output.split("\n"):
        data = _parse_json_line(line)
        if data and data.get("type") == "text":
            text = get_opencode_text(data)
            if text:
                texts.append(text)
    return "\n".join(texts) if texts else "(no text output extracted)"


def _find_assistant_text(messages: List[Dict[str, Any]]) -> Optional[str]:
    """在会话消息列表中查找 assistant 角色的文本回复"""
    for msg in reversed(messages):
        if msg.get("info", {}).get("role") == "assistant":
            parts = msg.get("parts", [])
            texts = [p["text"] for p in parts if p.get("type") == "text" and p.get("text")]
            if texts:
                return "\n".join(texts)
    return None


def extract_ai_text_from_session(session_file: str) -> Optional[str]:
    """从 opencode 导出的会话 JSON 中提取 AI 文本回复（流式输出提取失败的 fallback）"""
    try:
        with open(session_file, 'r', encoding='utf-8') as f:
            session = json.load(f)
        return _find_assistant_text(session.get("messages", []))
    except Exception:
        return None


def extract_reasoning(full_output: str) -> str:
    """从 full_output 的 JSON 事件中提取 AI 思考过程和工具调用"""
    parts = []
    for line in full_output.split("\n"):
        data = _parse_json_line(line)
        if not data:
            continue
        if data.get("type") == "tool_use":
            tool = data.get("part", {}).get("tool", "")
            state = data.get("part", {}).get("state", {})
            status = state.get("status", "completed")
            note = ""
            if status == "failed":
                note = " ⚠️ 该失败可能是沙箱环境/权限配置问题，非 skill 自身质量缺陷"
            parts.append(f"[调用工具: {tool}, 状态: {status}]{note}")
        elif data.get("type") == "reasoning":
            text = get_opencode_text(data)
            if text:
                parts.append(f"[思考] {text[:500]}")
    return "\n".join(parts) if parts else "(无工具调用)"


REVIEW_RUBRIC = """
## 评分标准（总分 100，总分 ≥ 60 且各维度均不低于最低阈值方为通过）

### 信息覆盖度（0-40 分）— 最低通过阈值：20 分
逐项检查预期输出中的每个关键要点是否被覆盖：
- 预期输出中通常以数字编号列出了关键要点（如 "1. xxx 2. xxx"），请逐条检查
- 每个要点满分 10 分（当预期有 4 个要点时），按覆盖程度打分：
  * 10 分：完整覆盖该要点
  * 5-9 分：部分覆盖，有缺失
  * 0-4 分：基本未覆盖或严重遗漏
- 若预期输出未明确编号，则按语义要点自行划分并评分
- 计算公式：各要点得分求和 = 覆盖度总分（上限 40）

### 技术准确性（0-30 分）— 最低通过阈值：15 分
逐项检查技术内容的正确性：
1. 技术术语使用正确（10 分）：每处术语错误扣 3 分，扣完为止
2. API / 命令 / 参数引用准确（10 分）：每处编造或错误引用扣 3 分
3. 逻辑流程正确、无自相矛盾（10 分）：每处矛盾扣 5 分

### 回复质量（0-20 分）— 最低通过阈值：10 分
逐项检查回复的可读性和实用性：
1. 结构清晰、分段合理（5 分）：混乱扣 2-3 分
2. 包含可操作的代码示例或具体步骤（5 分）：纯理论无示例扣 3-5 分
3. 直接回应用户问题，无跑题（5 分）：跑题扣 3-5 分
4. 中文表达流畅、专业（5 分）：严重语法/表达问题扣 2-3 分

### Token 消耗（0-10 分）— 最低通过阈值：3 分
逐项检查效率：
1. 回复长度合理，无冗余啰嗦（5 分）：明显冗余扣 2-3 分
2. 思考过程中的工具调用必要且高效（5 分）：无意义的重复工具调用扣 2-3 分
   注意：工具调用因环境/配置问题导致的失败（如带有 ⚠️ 标记的 read error、
   权限错误等），属于沙箱环境问题，不属于 skill 质量缺陷，不应扣分。

## 通过规则
1. 总分 ≥ 60
2. **四个维度均不低于各自的最低阈值**
3. 任一维度不达标，应判定为不通过（status: "fail"）

## 评审注意事项
- 不要求逐字匹配，语义覆盖即可
- 额外提供的合理信息不扣分
- reason 字段必须填写，逐一说明各维度的得分和扣分依据
"""


@dataclass
class ReviewPromptContext:
    """封装 create_review_prompt 的评审参数"""
    original_prompt: str
    ai_response: str
    reasoning: str
    expected_output: str
    eval_mode: str = "text"
    file_list: Optional[List[str]] = None
    sandbox_path: Optional[Path] = None
    dim_thresholds: Optional[Dict[str, int]] = None


def create_review_prompt(ctx: ReviewPromptContext) -> str:
    """构造评测 session 的完整 prompt（评分标准 + 动态数据 + 模板填写指引）。

    评审 Agent 不再输出 JSON，而是读取并填写沙箱中的 review-template.md 文件。
    """
    file_section = ""
    if ctx.eval_mode == "file_based" and ctx.file_list:
        file_paths = "\n".join(f"- {f}" for f in ctx.file_list)
        file_section = f"""
### 生成的文件清单（路径基于沙箱目录）
{file_paths}

## 评测流程
1. 使用 Read 工具逐一读取以上文件清单中的每个文件（文件在沙箱目录中，直接使用相对路径读取即可）
2. 根据文件实际内容，结合预期输出要点，进行评分
3. 文件的代码质量、结构完整性、正确性和可读性都应作为评分依据
"""
    # 若有非默认的维度阈值覆盖，注入到 prompt 中告知评审 AI
    threshold_note = ""
    if ctx.dim_thresholds:
        parts = []
        for dim in DIMENSION_ORDER:
            if dim in ctx.dim_thresholds:
                max_ = DIMENSION_MAX_SCORES.get(dim, "?")
                parts.append(f"- {dim}: >= {ctx.dim_thresholds[dim]}（满分 {max_}）")
        if parts:
            threshold_note = "\n### 本用例的维度阈值（覆盖默认值）\n" + "\n".join(parts) + "\n"

    if ctx.eval_mode == "file_based" and ctx.file_list:
        header = "你是一个技能测试评审员。请对以下 AI 对话和生成的文件进行评分和评审。"
    else:
        header = "你是一个技能测试评审员。请对以下 AI 对话进行评分和评审。"
    return f"""{header}
{REVIEW_RUBRIC}
{threshold_note}## 待评审对话

### 用户原始问题
{ctx.original_prompt}

### AI 的思考过程与工具调用
{ctx.reasoning}

### AI 的最终回复
{ctx.ai_response}

### 预期回复应覆盖的要点
{ctx.expected_output}
{file_section}
## 评审操作步骤
1. 使用 Read 工具读取当前目录下的 review-template.md 文件
2. 根据上述评分标准进行全面评审
3. 使用 Write 工具将评审结果写入 review-template.md，替换所有方括号占位符：
   - [PASS/FAIL] 替换为实际判定结果（PASS 或 FAIL）
   - [0-100] 替换为实际总分（0-100 的整数）
   - 表格中 [0-40]、[0-30]、[0-20]、[0-10] 替换为各维度实际得分（整数）
   - 表格中 [YES/NO] 替换为各维度是否通过（得分 >= 阈值则为 YES，否则为 NO）
   - [detailed review text here] 替换为详细评审意见，逐一说明每个维度的得分理由和扣分依据
4. 只替换方括号占位符及其内容，不要修改模板的其他结构（标题、表格、分隔线、HTML注释）"""


def _run_review_session(
        opencode_runner: OpencodeRunner,
        review_prompt: str,
        session_name: str,
) -> tuple:
    """运行评测 session 并返回 (review_lines, review_error)"""
    review_lines = []
    review_error = ""

    for chunk in opencode_runner.run_streaming(
            prompt=review_prompt,
            skill=".",
            session_name=f"{session_name}_review"
    ):
        chunk_type = chunk.get("type")
        if chunk_type == "json_output":
            data = chunk.get("data", {})
            review_lines.append(json.dumps(data, ensure_ascii=False))
        elif chunk_type == "raw_output":
            review_lines.append(chunk.get("data", ""))
        elif chunk_type == "text":
            review_lines.append(chunk.get("text", ""))
        elif chunk_type in ("error", "exception"):
            review_error = chunk.get("data", "")
            logger.error("[REVIEW %s] %s", chunk_type.upper(), review_error)

    return review_lines, review_error


def _check_contains_pattern(
        full_output: str, ai_text: str, pattern: str,
        eval_id: Optional[str], truncate_len: int) -> None:
    """检查输出是否包含指定模式（搜索 full_output）"""
    if pattern not in full_output:
        pytest.fail(
            f"[contains] 期望输出中包含 \"{pattern}\"，但未找到。\n"
            f"--- AI 回复 ---\n"
            f"{ai_text[:truncate_len]}\n"
            f"--- 结束 ---"
        )


def _check_not_contains_pattern(
        full_output: str, ai_text: str, pattern: str,
        eval_id: Optional[str], truncate_len: int) -> None:
    """检查 AI 最终回复不应包含指定模式。

    只检查 ai_text（AI 对用户的最终回复），不检查 full_output。
    full_output 中包含 skill 加载时的参考文档内容（如 references/quick_ref.md），
    这些文档中的术语不应被视为 AI 的回复内容。
    """
    if pattern in ai_text:
        pytest.fail(
            f"[not_contains] 期望 AI 回复中不包含 \"{pattern}\"，但实际出现了。\n"
            f"--- AI 回复 ---\n"
            f"{ai_text[:truncate_len]}\n"
            f"--- 结束 ---"
        )


def _check_file_exists(
        skill_dir: Optional[Path], path: str, eval_id: Optional[str],
        sandbox_path: Optional[Path] = None) -> None:
    """检查文件是否存在
    搜索顺序：sandbox/<path> → sandbox/skill/<path> → skill_dir/<path> → 递归搜索 sandbox
    """
    candidates = []
    if sandbox_path:
        candidates.append(sandbox_path / path)
        if skill_dir:
            candidates.append(sandbox_path / ".opencode" / "skills" / skill_dir.name / path)
    if skill_dir:
        candidates.append(skill_dir / path)
    for fp in candidates:
        if fp.exists():
            return
    # 递归搜索 fallback：AI 可能将文件写入子目录（如 operators/Add/docs/ 等）
    if sandbox_path:
        matches = list(sandbox_path.rglob(path))
        if matches:
            logger.info(
                "[file_exists] e%s 通过递归搜索找到文件: %s",
                eval_id or "?", matches[0],
            )
            return
    pytest.fail(
        f"[file_exists] 期望文件 \"{path}\" 未找到。"
        f"搜索路径: {[str(c) for c in candidates]}"
    )


def _check_file_list(
        sandbox_path: Optional[Path], pattern: str, eval_id: Optional[str]) -> None:
    """检查沙箱中是否存在匹配 glob pattern 的文件
    搜索顺序：sandbox → sandbox/skill/
    """
    if not sandbox_path:
        raise ValueError("sandbox_path is required for file_list expectation")
    candidates = [sandbox_path]
    # 同时搜索 .opencode/skills/ 下的各 skill 子目录
    skills_dir = sandbox_path / ".opencode" / "skills"
    if skills_dir.exists():
        for skill_subdir in skills_dir.iterdir():
            if skill_subdir.is_dir():
                candidates.append(skill_subdir)
    for base in candidates:
        if base.exists():
            matches = list(base.glob(pattern))
            if matches:
                return
    pytest.fail(
        f"[file_list] 未找到匹配 glob \"{pattern}\" 的文件。"
        f"搜索范围: 沙箱及 skill 子目录"
    )


def _parse_file_contains_pattern(raw: str) -> Tuple[str, List[str]]:
    """解析 [file_contains] 的模式串。

    输入: 'operators/add_vector/op_kernel/*.asc : "__global__";"LocalTensor"'
    输出: ('operators/add_vector/op_kernel/*.asc', ['__global__', 'LocalTensor'])
    """
    parts = raw.split(" : ", 1)
    if len(parts) != 2:
        raise ValueError(
            f"[file_contains] 格式错误，需要 \"path : \\\"p1\\\";\\\"p2\\\"\": {raw}"
        )
    file_path = parts[0].strip()
    patterns = re.findall(r'"([^"]*)"', parts[1])
    if not patterns:
        raise ValueError(f"[file_contains] 未提取到任何文本模式: {raw}")
    return file_path, patterns


def _check_file_contains(
        sandbox_path: Optional[Path], file_pattern: str,
        patterns: List[str], eval_id: Optional[str]) -> None:
    """检查沙箱中匹配 glob 的文件是否至少有一个包含所有文本模式。

    支持 glob 通配符（*、?、[]），匹配到多个文件时，
    只要有一个文件包含所有 pattern 即判定通过。
    """
    if not sandbox_path:
        raise ValueError("sandbox_path is required for file_contains expectation")

    abs_pattern = str(sandbox_path / file_pattern)
    matching_files = glob_module.glob(abs_pattern)

    if not matching_files:
        pytest.fail(
            f"[file_contains] 未找到匹配 \"{file_pattern}\" 的文件。\n"
            f"搜索路径: {abs_pattern}"
        )

    for fp_str in matching_files:
        try:
            content = Path(fp_str).read_text(encoding="utf-8")
        except Exception:
            logger.warning("无法读取文件 %s，跳过", fp_str)
            continue
        if all(p in content for p in patterns):
            return  # 找到包含所有模式的文件，通过

    matched_names = [str(Path(p).name) for p in matching_files]
    pytest.fail(
        f"[file_contains] 匹配的文件中未找到所有指定文本。\n"
        f"匹配文件: {matched_names}\n"
        f"目标模式: {patterns}"
    )


def _load_ses_data(ses_path: Path) -> Dict[str, Any]:
    """读取导出的 session JSON；解析失败直接 pytest.fail"""
    try:
        with open(ses_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        pytest.fail(
            f"[skill_activated] 无法读取 session 导出文件 '{ses_path}': {e}"
        )
        return {}  # 不可达，仅为满足静态检查（pytest.fail 会 raise）


def _extract_skill_name_from_part(part: Dict[str, Any]) -> str:
    """从单个 message part 中提取 skill 工具调用的 name 字段"""
    if part.get("type") != "tool" or part.get("tool") != "skill":
        return ""
    state = part.get("state", {})
    return (state.get("input", {}).get("name", "")
            or state.get("metadata", {}).get("name", ""))


def _collect_activated_skills(ses_data: Dict[str, Any]) -> List[str]:
    """从 session JSON 收集所有实际加载的 skill 名称"""
    activated: List[str] = []
    for msg in ses_data.get("messages", []):
        for part in msg.get("parts", []):
            name = _extract_skill_name_from_part(part)
            if name:
                activated.append(name)
    return activated


def _fail_skill_not_activated(eval_tag: str, expected: str,
                              activated: List[str]) -> None:
    """根据是否有加载记录构造清晰的失败消息"""
    if activated:
        skill_list = "\n    - ".join(activated)
        pytest.fail(
            f"{eval_tag}正向看护失败：期望激活 skill \"{expected}\"，"
            f"但 AI 实际加载了以下 skill：\n"
            f"    - {skill_list}\n"
            f"请检查 prompt 是否与目标 skill 的触发条件匹配，"
            f"或干扰 skill 列表是否过于相似。"
        )
    pytest.fail(
        f"{eval_tag}正向看护失败：期望激活 skill \"{expected}\"，"
        f"但 AI 没有加载任何 skill。"
        f"请检查 prompt 是否能触发目标 skill。"
    )


def _stream_event_matches_skill(data: Dict[str, Any], expected: str) -> bool:
    """旧流式格式兼容：单个事件是否命中目标 skill"""
    if data.get("type") == "tool_use" and data.get("part", {}).get("tool", "").lower() == "skill":
        state = data.get("part", {}).get("state", {})
        return any(expected in str(v) for v in state.get("input", {}).values())
    if data.get("type") == "tool" and data.get("tool") == "skill":
        return data.get("state", {}).get("input", {}).get("name", "") == expected
    return False


def _scan_stream_fallback(full_output: str, expected: str) -> bool:
    """从流式输出搜索目标 skill 加载事件（兼容旧格式）"""
    for line in full_output.split("\n"):
        data = _parse_json_line(line)
        if data and _stream_event_matches_skill(data, expected):
            return True
    return False


def _validate_skill_activated(ctx: ExpectationContext) -> None:
    """程序化检查目标 skill 是否被 AI 加载（解析导出 session JSON）

    从导出的 session JSON（ses.json）中遍历所有 message parts，
    查找 type=tool, tool=skill 的事件，然后匹配 state.input.name。
    不依赖 AI 评审模型，提供确定性判断。
    """
    expected = ctx.exp.get("pattern", "")
    if not expected:
        pytest.fail("[skill_activated] 缺少 pattern（期望的 skill 名称）")

    ses_path = ctx.session_export_path
    activated: List[str] = []
    if ses_path and ses_path.exists():
        activated = _collect_activated_skills(_load_ses_data(ses_path))
        if expected in activated:
            return
        if activated:
            # AI 加载了其他 skill 而非目标 skill — 立即失败
            eval_tag = f"(Eval {ctx.eval_id}) " if ctx.eval_id else ""
            _fail_skill_not_activated(eval_tag, expected, activated)

    if _scan_stream_fallback(ctx.full_output, expected):
        return

    eval_tag = f"(Eval {ctx.eval_id}) " if ctx.eval_id else ""
    if activated:
        _fail_skill_not_activated(eval_tag, expected, activated)
    else:
        pytest.fail(
            f"{eval_tag}正向看护失败：期望激活 skill \"{expected}\"，"
            f"但 AI 没有加载任何 skill。"
            f"请检查 prompt 是否能触发目标 skill。"
        )


def _validate_expectation(ctx: ExpectationContext) -> None:
    """验证单个期望"""
    exp_type = ctx.exp.get("type")
    if exp_type == "contains":
        _check_contains_pattern(ctx.full_output, ctx.ai_text, ctx.exp.get("pattern", ""), ctx.eval_id, ctx.truncate_len)
    elif exp_type == "file_exists":
        _check_file_exists(ctx.skill_dir, ctx.exp.get("pattern", ""), ctx.eval_id, ctx.sandbox_path)
    elif exp_type == "file_list":
        _check_file_list(ctx.sandbox_path, ctx.exp.get("pattern", ""), ctx.eval_id)
    elif exp_type == "not_contains":
        _check_not_contains_pattern(
            ctx.full_output, ctx.ai_text, ctx.exp.get("pattern", ""),
            ctx.eval_id, ctx.truncate_len
        )
    elif exp_type == "skill_activated":
        _validate_skill_activated(ctx)
    elif exp_type == "file_contains":
        file_path, patterns = _parse_file_contains_pattern(ctx.exp.get("pattern", ""))
        _check_file_contains(ctx.sandbox_path, file_path, patterns, ctx.eval_id)


def validate_output(ctx: ValidationContext) -> None:
    """
    通用输出验证函数
    """
    if not ctx.ai_text:
        ctx.ai_text = extract_ai_text(ctx.full_output)

    if ctx.expected_output:
        _validate_expected_output(ctx)

    if ctx.expectations:
        ses_path = ctx.opencode_runner.session_dir / f"{ctx.session_name}_ses.json"
        for exp in ctx.expectations:
            exp_ctx = ExpectationContext(
                exp=exp,
                full_output=ctx.full_output,
                ai_text=ctx.ai_text,
                skill_dir=ctx.skill_dir,
                sandbox_path=ctx.sandbox_path,
                eval_id=ctx.eval_id,
                truncate_len=ctx.truncate_len,
                skill_name=ctx.skill_name,
                session_export_path=ses_path,
            )
            _validate_expectation(exp_ctx)


def _copy_review_template(sandbox_path: Optional[Path]) -> Optional[Path]:
    """将 review-template.md 复制到沙箱目录，返回目标路径或 None。"""
    template_src = FRAMEWORK_DIR / "config" / "review-template.md"
    template_dst = sandbox_path / "review-template.md" if sandbox_path else None

    if template_src.exists() and template_dst:
        shutil.copy2(template_src, template_dst)
        return template_dst
    logger.warning("[TEMPLATE] review-template.md not found at %s", template_src)
    return None


def _read_review_result(template_dst: Optional[Path]) -> Dict[str, Any]:
    """从沙箱读取填写后的 review-template.md 并解析评审结果。"""
    if template_dst and template_dst.exists():
        try:
            review_content = template_dst.read_text(encoding="utf-8")
            return parse_review_md(review_content)
        except (IOError, OSError) as e:
            return {"status": "error", "reason": f"读取模板文件失败: {e}"}
    return {"status": "error", "reason": "评审后沙箱中未找到 review-template.md"}


def _assert_review_passed(result: Dict[str, Any], ctx: ValidationContext) -> None:
    """校验评审结果：检查 reason 存在性、维度阈值和 pass/fail 状态。"""
    if not result.get("reason"):
        assert False, (
            f"Eval {ctx.eval_id}: review result missing reason | "
            f"Review output: {json.dumps(result, ensure_ascii=False)}"
        )

    dim_thresholds = ctx.dim_thresholds or DEFAULT_DIMENSION_THRESHOLDS
    dim_scores = parse_dimension_scores(result.get("dimensions"))
    dim_check_msg = validate_dimension_scores(dim_scores, dim_thresholds, ctx.eval_id or "")

    if result.get("status") == "pass" and dim_check_msg is None:
        return
    reasons = [result.get("reason", "unknown")] if result.get("status") != "pass" else []
    if dim_check_msg:
        reasons.append(dim_check_msg)
    full_reason = " | ".join(reasons)
    msg = f"Eval {ctx.eval_id}: expected_output check failed - {full_reason[:200]}"
    assert False, msg


def _validate_expected_output(ctx: ValidationContext) -> None:
    """验证 AI 回复是否符合预期输出（基于 MD 模板评审）。

    流程:
    1. 提取 AI 思考过程
    2. 复制模板到沙箱 → 构建评审 prompt → 运行评审 session
    3. 读取填写后的 MD 模板 → parse_review_md() → 程序化阈值校验
    """
    reasoning = extract_reasoning(ctx.full_output)

    template_dst = _copy_review_template(ctx.sandbox_path)

    review_prompt = create_review_prompt(ReviewPromptContext(
        original_prompt=ctx.original_prompt,
        ai_response=ctx.ai_text[:ctx.truncate_len],
        reasoning=reasoning[:ctx.truncate_len],
        expected_output=ctx.expected_output,
        eval_mode=ctx.eval_mode,
        file_list=ctx.generated_files if ctx.eval_mode == "file_based" else None,
        sandbox_path=ctx.sandbox_path if ctx.eval_mode == "file_based" else None,
        dim_thresholds=ctx.dim_thresholds,
    ))

    review_lines, review_error = _run_review_session(
        ctx.opencode_runner, review_prompt, ctx.session_name
    )
    assert not review_error, f"Eval {ctx.eval_id}: review session error - {review_error}"

    result = _read_review_result(template_dst)
    brief = {k: v for k, v in result.items() if k != "reason"}
    logger.info("[REVIEW RESULT] %s", json.dumps(brief, ensure_ascii=False))
    _assert_review_passed(result, ctx)


def _resolve_distractor_dirs(distractor_names: List[str], skill: str,
                              eval_id: Any) -> List[Path]:
    """根据干扰 skill 名解析其目录，缺失项打 warning 跳过"""
    dirs: List[Path] = []
    for ds_name in distractor_names:
        ds_dir = get_skill_path(ds_name)
        if ds_dir:
            dirs.append(ds_dir)
        else:
            logger.warning(
                "Distractor skill '%s' not found for %s::eval_%s, skipping",
                ds_name, skill, eval_id,
            )
    return dirs


def _build_eval_test_case(skill: str, skill_dir: Optional[Path],
                          eval_item: Dict[str, Any]) -> Dict[str, Any]:
    """从单条 eval 用例构造 pytest 参数化字典"""
    distractor_dirs = _resolve_distractor_dirs(
        eval_item.get("distractor_skills", []) or [],
        skill, eval_item.get("id"),
    )
    return {
        "skill_name": skill,
        "eval_mode": eval_item.get("eval_mode", "text"),
        "eval": eval_item,
        "skill_dir": skill_dir,
        "distractor_skill_dirs": distractor_dirs,
    }


def pytest_generate_tests(metafunc):
    if "eval_case" not in metafunc.fixturenames:
        return

    skill_names = metafunc.config.getoption("--skill", None)
    eval_id = metafunc.config.getoption("--eval-id", None)
    ascend_platforms = metafunc.config.getoption("--ascend-platform", None)

    test_cases: List[Dict[str, Any]] = []
    ids: List[str] = []

    for skill in get_skills_with_evals():
        if skill_names and skill not in skill_names:
            continue
        evals_data = load_evals_md(skill)
        if not evals_data:
            continue
        skill_dir = get_skill_path(skill)
        for eval_item in evals_data.get("evals", []):
            if eval_id and str(eval_item.get("id")) != str(eval_id):
                continue
            if not _platform_matches(ascend_platforms, eval_item):
                continue
            test_cases.append(_build_eval_test_case(skill, skill_dir, eval_item))
            ids.append(f"{skill}::eval_{eval_item.get('id')}")

    if test_cases:
        metafunc.parametrize("eval_case", test_cases, ids=ids, scope="function")
    else:
        # 无匹配用例时，生成一个占位用例并立即跳过，避免 fixture 未定义的 ERROR
        dummy = {"_skip": True}
        metafunc.parametrize("eval_case", [dummy], ids=["no-matching-cases"], scope="function")


def _log_eval_case_header(skill_name: str, eval_id: Any, prompt: str,
                           expected_output: str,
                           distractor_skill_dirs: Optional[List[Path]] = None) -> None:
    """打印评测用例执行前的日志头"""
    logger.info("=" * 60)
    logger.info("[%s] 评测用例 %s 开始执行", skill_name, eval_id)
    if distractor_skill_dirs:
        logger.info("[%s] 干扰技能 (%d): %s", skill_name,
                    len(distractor_skill_dirs),
                    ", ".join(d.name for d in distractor_skill_dirs))
    logger.info("=" * 60)


def _collect_exec_output(
        opencode_runner: OpencodeRunner,
        prompt: str, skill_ref: Optional[str], session_name: str
) -> tuple:
    """运行 exec session 并收集输出，返回 (full_output, error_output, session_file, success)"""
    output_lines = []
    error_output = ""
    session_file = None
    success = True

    for chunk in opencode_runner.run_stream(
            prompt=prompt,
            skill=skill_ref,
            session_name=session_name
    ):
        chunk_type = chunk.get("type")

        if chunk_type == "json_output":
            data = chunk.get("data", {})
            output_lines.append(json.dumps(data, ensure_ascii=False))
        elif chunk_type == "raw_output":
            output_lines.append(chunk.get("data", ""))
        elif chunk_type == "error":
            error_msg = chunk.get("data", "")
            logger.error("[ERROR] %s", error_msg)
            error_output = error_msg
            success = False
        elif chunk_type == "complete":
            session_file = chunk.get("session_file")
            success = chunk.get("returncode", 0) == 0
        elif chunk_type == "exception":
            error_output = chunk.get("data", "")
            logger.error("[EXCEPTION] %s", error_output)
            success = False

    return "\n".join(output_lines), error_output, session_file, success


def _setup_eval_sandbox(sandbox_manager: SandboxManager, inputs: _EvalInputs):
    """创建沙箱和 opencode runner。

    使用 _EvalInputs 封装参数以减少函数签名（≤5 参数要求）。
    """
    sandbox_path = sandbox_manager.create_sandbox(inputs.skill_name, inputs.eval_id)
    sandbox_manager.create_skill_link(sandbox_path, inputs.skill_dir)
    for ds_dir in (inputs.distractor_skill_dirs or []):
        sandbox_manager.create_skill_link(sandbox_path, ds_dir)
    opencode_runner = create_opencode_runner(
        sandbox_manager, sandbox_path,
        timeout=inputs.eval_data.get("timeout"),
    )
    return opencode_runner, sandbox_path


def _create_and_validate(opencode_runner, session_name, full_output,
                         inputs: _EvalInputs, ai_text, sandbox_path):
    """创建 ValidationContext 并执行验证（消除 skill/team 重复代码）。"""
    ctx = ValidationContext(
        opencode_runner=opencode_runner,
        session_name=session_name,
        full_output=full_output,
        original_prompt=inputs.prompt,
        expected_output=inputs.expected_output,
        expectations=inputs.expectations,
        skill_dir=inputs.skill_dir,
        eval_id=inputs.eval_id,
        ai_text=ai_text,
        eval_mode=inputs.eval_mode,
        sandbox_path=sandbox_path,
        generated_files=collect_generated_files(sandbox_path,
                                                original_skill_dir=inputs.skill_dir),
        skill_name=inputs.skill_name,
        dim_thresholds=inputs.eval_data.get("dim_thresholds"),
        truncate_len=inputs.eval_data.get("truncate_len", 30000),
    )
    validate_output(ctx)


def _detect_model_from_session(ses_file: str) -> Optional[str]:
    """从导出的 session JSON 中自动检测实际使用的模型名称"""
    try:
        with open(ses_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        model_id = data.get("info", {}).get("model", {}).get("id")
        if model_id:
            logger.info("[自动检测] 当前 session 使用的模型: %s", model_id)
        return model_id
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def _check_token_budget(eval_data: Dict[str, Any], eval_id, opencode_runner,
                        session_name: str) -> None:
    """检查 token 消耗是否超过硬性阈值"""
    max_tokens = eval_data.get("max_tokens")

    # 始终从 session 导出数据自动检测实际使用的模型名称，
    # 按模型匹配专用的 Max Tokens (<model>)。
    # 检测到的模型是 opencode 实际使用的模型（如 deepseek-v4-flash-free），
    # 可能与 --eval-model 传入的名称不同（opencode 可能解析为具体版本）。
    ses_file = str(opencode_runner.session_dir / f"{session_name}_ses.json")
    model_name = _detect_model_from_session(ses_file)
    if model_name:
        per_model = eval_data.get("max_tokens_by_model", {})
        if model_name in per_model:
            max_tokens = per_model[model_name]

    if max_tokens is None:
        return
    from session_stats import SessionStats
    ses_file = str(opencode_runner.session_dir / f"{session_name}_ses.json")
    stats = SessionStats.from_export_file(ses_file)
    actual_tokens = stats.tokens.total
    assert actual_tokens <= max_tokens, (
        f"Eval {eval_id}: token 消耗 ({actual_tokens}) 超过上限 ({max_tokens})"
    )


@dataclass
class _EvalInputs:
    """test_eval_case 解构后的输入参数"""
    skill_name: str
    eval_id: Any
    prompt: str
    expected_output: str
    expectations: List[Dict[str, Any]]
    eval_mode: str
    skill_dir: Optional[Path]
    distractor_skill_dirs: List[Path]
    eval_data: Dict[str, Any]


def _unpack_eval_inputs(eval_case: Dict[str, Any]) -> _EvalInputs:
    """从 pytest 参数化字典中抽出测试需要的字段，并完成 prompt 安全校验"""
    eval_data = eval_case["eval"]
    eval_id = eval_data.get("id")
    prompt = eval_data.get("prompt", "")
    assert prompt, f"Eval {eval_id}: prompt is required"
    _validate_prompt(prompt, str(eval_id))

    eval_mode = eval_case.get("eval_mode", "text")
    if eval_mode == "file_based":
        prompt = prompt.rstrip() + FILE_BASED_HINT

    return _EvalInputs(
        skill_name=eval_case["skill_name"],
        eval_id=eval_id,
        prompt=prompt,
        expected_output=eval_data.get("expected_output", ""),
        expectations=eval_data.get("expectations", []),
        eval_mode=eval_mode,
        skill_dir=eval_case["skill_dir"],
        distractor_skill_dirs=eval_case.get("distractor_skill_dirs", []),
        eval_data=eval_data,
    )


def _run_and_extract_text(opencode_runner, prompt: str, cwd: str,
                         skill_name: str, eval_id: str) -> tuple:
    """执行 opencode 并提取 AI 文本回复（含 fallback 逻辑）"""
    session_name = f"{skill_name}_case_{eval_id}"
    full_output, error_output, session_file, success = _collect_exec_output(
        opencode_runner, prompt, cwd, session_name,
    )
    assert success, f"Eval {eval_id}: opencode run failed - {error_output}"

    ai_text = extract_ai_text(full_output)

    session_export_path = str(opencode_runner.session_dir / f"{session_name}_ses.json")
    opencode_runner.export_session_data(output_file=session_export_path)

    # Fallback: 若从流式输出未能提取到文本，从已导出的会话 JSON 中提取
    # （不触发 skill 时 opencode 流式事件格式可能不含 type:text，导致 extract_ai_text 返回占位符）
    if "(no text output extracted)" in ai_text:
        session_text = extract_ai_text_from_session(session_export_path)
        if session_text:
            ai_text = session_text
            logger.warning("FALLBACK: 从会话导出文件中提取到 AI 文本 (eval %s)", eval_id)

    return full_output, session_name, session_file, ai_text


def _run_eval_with_retry(
    opencode_runner, sandbox_path: str, inputs: EvalInputs,
) -> None:
    """执行评测并支持重试，验证不通过时抛出异常。"""
    max_retries = int(os.environ.get("EVAL_EXEC_RETRIES", "1"))
    last_error = None
    best_session_file = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.warning("[RETRY %d/%d] 重试执行 eval %s",
                           attempt, max_retries, inputs.eval_id)

        try:
            full_output, session_name, session_file, ai_text = _run_and_extract_text(
                opencode_runner, inputs.prompt, ".",
                inputs.skill_name, str(inputs.eval_id),
            )
        except AssertionError as e:
            last_error = e
            logger.warning("[EXEC FAIL] opencode 执行失败 (attempt %d): %s",
                           attempt, str(e)[:200])
            continue

        _check_token_budget(inputs.eval_data, inputs.eval_id,
                            opencode_runner, session_name)

        try:
            _create_and_validate(opencode_runner, session_name, full_output,
                                 inputs, ai_text, sandbox_path)
            last_error = None
            best_session_file = session_file
            break  # 评测通过
        except AssertionError as e:
            last_error = e
            logger.warning("[VALIDATE FAIL] 评测不通过 (attempt %d): %s",
                           attempt, str(e)[:200])
            if not best_session_file:
                best_session_file = session_file

    if last_error is not None:
        raise last_error
    logger.info("Session file: %s", best_session_file)


def test_eval_case(eval_case: Dict[str, Any], sandbox_manager: SandboxManager):
    # 占位用例过滤：无匹配 case 时跳过
    if eval_case.get("_skip"):
        pytest.skip("No matching eval cases for the current filter")

    if os.environ.get("REPORT_ONLY") == "1":
        logger.info("[%s] REPORT_ONLY 模式，跳过测试执行 (eval %s)",
                    eval_case["skill_name"], eval_case["eval"].get("id"))
        return

    eval_data = eval_case["eval"]
    if eval_data.get("disabled"):
        pytest.skip(f"Eval {eval_data.get('id')} marked as Disabled - skipping")

    inputs = _unpack_eval_inputs(eval_case)

    opencode_runner, sandbox_path = _setup_eval_sandbox(sandbox_manager, inputs)
    _log_eval_case_header(inputs.skill_name, inputs.eval_id, inputs.prompt,
                          inputs.expected_output,
                          distractor_skill_dirs=inputs.distractor_skill_dirs)

    _run_eval_with_retry(opencode_runner, sandbox_path, inputs)
