# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

import pytest

from conftest import (
    get_skill_path, get_skills_with_evals, load_evals_md, REPO_ROOT,
    extract_review_json, get_opencode_text, strip_markdown_fence,
    FRAMEWORK_DIR, SANDBOX_DIR
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
    truncate_len: int = 2000
    # file_based 模式字段
    eval_mode: str = "text"
    sandbox_path: Optional[Path] = None
    generated_files: List[str] = field(default_factory=list)
    # 正向看护字段
    skill_name: str = ""


@dataclass
class ExpectationContext:
    """封装期望验证所需的参数"""
    exp: Dict[str, Any]
    full_output: str
    ai_text: str
    skill_dir: Optional[Path] = None
    sandbox_path: Optional[Path] = None
    eval_id: Optional[str] = None
    truncate_len: int = 2000
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
            parts.append(f"[调用工具: {tool}, 状态: {status}]")
        elif data.get("type") == "reasoning":
            text = get_opencode_text(data)
            if text:
                parts.append(f"[思考] {text[:500]}")
    return "\n".join(parts) if parts else "(无工具调用)"


REVIEW_RUBRIC = """
## 评分标准（总分 100，≥ 60 为通过）

### 信息覆盖度（0-40 分）
- 是否完整覆盖了预期回复中的所有关键要点
- 每遗漏一个重要要点扣 10-20 分

### 技术准确性（0-30 分）
- 技术信息是否正确，无错误或误导
- 命令、参数、版本号等信息是否准确

### 回复质量（0-20 分）
- 结构清晰、逻辑连贯
- 表达简洁、直接回应用户问题
- 无冗余或无关内容

### Token 消耗（0-10 分）
- 回复长度合理，无冗余啰嗦
- 思考过程中的工具调用是否必要、高效
- 过多冗余内容或无效工具调用应扣分

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


def create_review_prompt(ctx: ReviewPromptContext) -> str:
    """构造评测 session 的完整 prompt（评分机制 + 动态数据）"""
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
    if ctx.eval_mode == "file_based" and ctx.file_list:
        header = "你是一个技能测试评审员。请对以下 AI 对话和生成的文件进行评分和评审。"
    else:
        header = "你是一个技能测试评审员。请对以下 AI 对话进行评分和评审。"
    return f"""{header}
{REVIEW_RUBRIC}
## 待评审对话

### 用户原始问题
{ctx.original_prompt}

### AI 的思考过程与工具调用
{ctx.reasoning}

### AI 的最终回复
{ctx.ai_response}

### 预期回复应覆盖的要点
{ctx.expected_output}
{file_section}
请以 JSON 格式回复（只输出 JSON，不要其他内容）。重要：JSON 字符串值内如需使用双引号，必须用反斜杠转义（例如 \"示例\"），以免 JSON 解析失败。
{{"status": "pass", "score": 85, "reason": "覆盖度(40/40): 完整覆盖预期要点; 准确性(25/30): ...; 质量(12/20): 回复简洁; Token(8/10): 无冗余"}}
或
{{"status": "fail", "score": 35, "reason": "覆盖度(10/40): 遗漏要点...; 准确性(10/30): ...; 质量(10/20): ...; Token(5/10): ..."}}"""


def _extract_text_from_line(line: str) -> str:
    """从单行 JSON 中提取文本"""
    data = _parse_json_line(line)
    if not data:
        return ""
    return get_opencode_text(data) or ""


def parse_check_response(lines: List[str]) -> Dict[str, Any]:
    """从评测 session 的输出行中提取 status/reason，兼容 markdown 代码块包裹"""
    # 第一轮：按原有方式逐行解析
    for line in reversed(lines):
        text = _extract_text_from_line(line)
        if text and ("pass" in text or "fail" in text):
            result = extract_review_json(text)
            if result:
                return result

    # 第二轮：汇总所有文本行后整体搜索 JSON
    all_text = ""
    for line in reversed(lines):
        text = _extract_text_from_line(line)
        if text:
            all_text = text + "\n" + all_text

    if all_text:
        result = extract_review_json(all_text)
        if result:
            return result

    return {"status": "error", "reason": "无法从评测 session 输出中解析判定结果"}


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
    """检查输出是否包含指定模式"""
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
    """检查输出不应包含指定模式"""
    if pattern in full_output:
        pytest.fail(
            f"[not_contains] 期望输出中不包含 \"{pattern}\"，但实际出现了。\n"
            f"--- AI 回复 ---\n"
            f"{ai_text[:truncate_len]}\n"
            f"--- 结束 ---"
        )


def _check_file_exists(
        skill_dir: Optional[Path], path: str, eval_id: Optional[str],
        sandbox_path: Optional[Path] = None) -> None:
    """检查文件是否存在
    搜索顺序：sandbox/<path> → sandbox/skill/<path> → skill_dir/<path>
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
    if data.get("type") == "tool_use" and data.get("part", {}).get("tool", "") == "Skill":
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
    if ses_path and ses_path.exists():
        activated = _collect_activated_skills(_load_ses_data(ses_path))
        if expected in activated:
            return
        eval_tag = f"(Eval {ctx.eval_id}) " if ctx.eval_id else ""
        _fail_skill_not_activated(eval_tag, expected, activated)

    if _scan_stream_fallback(ctx.full_output, expected):
        return

    pytest.fail(
        f"正向看护失败：期望激活 skill \"{expected}\"，"
        f"但在 session 输出中未找到任何 skill 加载事件。"
    )


def _validate_expectation(ctx: ExpectationContext) -> None:
    """验证单个期望"""
    exp_type = ctx.exp.get("type")
    if exp_type == "contains":
        _check_contains_pattern(ctx.full_output, ctx.ai_text, ctx.exp.get("pattern", ""), ctx.eval_id, ctx.truncate_len)
    elif exp_type == "file_exists":
        _check_file_exists(ctx.skill_dir, ctx.exp.get("path", ""), ctx.eval_id, ctx.sandbox_path)
    elif exp_type == "file_list":
        _check_file_list(ctx.sandbox_path, ctx.exp.get("pattern", ""), ctx.eval_id)
    elif exp_type == "not_contains":
        _check_not_contains_pattern(
            ctx.full_output, ctx.ai_text, ctx.exp.get("pattern", ""),
            ctx.eval_id, ctx.truncate_len
        )
    elif exp_type == "skill_activated":
        _validate_skill_activated(ctx)


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


def _try_extract_review_result(text: str) -> Optional[Dict[str, Any]]:
    """尝试从文本中提取评审结果，兼容 opencode 双编码 JSON"""
    result = extract_review_json(text)
    if result:
        return result
    unescaped = text.replace('\\"', '"')
    if unescaped != text:
        return extract_review_json(unescaped)
    return None


def _parse_review_from_export(export_data: dict) -> Dict[str, Any]:
    """从导出的 review session 数据中解析评审结果"""
    messages = export_data.get("messages", [])
    for msg in reversed(messages):
        for part in msg.get("parts", []):
            if part.get("type") != "text":
                continue
            text = part.get("text", "")
            if not text or not ("pass" in text.lower() or "fail" in text.lower()):
                continue
            result = _try_extract_review_result(text)
            if result:
                return result

    all_text = ""
    for msg in reversed(messages):
        for part in msg.get("parts", []):
            if part.get("type") != "text":
                continue
            t = part.get("text", "")
            if t:
                all_text = t + "\n" + all_text
    if all_text:
        result = extract_review_json(all_text)
        if result:
            return result

    return {"status": "error", "reason": "无法从导出文件中解析判定结果"}


def _validate_expected_output(ctx: ValidationContext) -> None:
    """验证 AI 回复是否符合预期输出"""
    reasoning = extract_reasoning(ctx.full_output)

    logger.debug("--- AI REASONING ---")
    logger.debug(reasoning[:2000] if reasoning else "(无思考过程)")
    logger.debug("--- END AI REASONING ---")

    review_prompt = create_review_prompt(ReviewPromptContext(
        original_prompt=ctx.original_prompt,
        ai_response=ctx.ai_text[:ctx.truncate_len],
        reasoning=reasoning[:ctx.truncate_len],
        expected_output=ctx.expected_output,
        eval_mode=ctx.eval_mode,
        file_list=ctx.generated_files if ctx.eval_mode == "file_based" else None,
        sandbox_path=ctx.sandbox_path if ctx.eval_mode == "file_based" else None,
    ))

    logger.debug("--- REVIEW PROMPT ---")
    logger.debug(review_prompt)
    logger.debug("--- END REVIEW PROMPT ---")

    review_lines, review_error = _run_review_session(
        ctx.opencode_runner, review_prompt, ctx.session_name
    )

    assert not review_error, f"Eval {ctx.eval_id}: review session error - {review_error}"

    export_file = str(ctx.opencode_runner.session_dir / f"{ctx.session_name}_review_ses.json")
    export_result = ctx.opencode_runner.export_session_data(output_file=export_file)

    # 优先从导出数据解析（更稳定），流式行作为回退
    review_data = export_result.get("data") if export_result.get("success") else None
    if review_data:
        result = _parse_review_from_export(review_data)
    else:
        result = {"status": "error", "reason": "export_session_data 失败"}

    if result.get("status") == "error":
        logger.info("[REVIEW] export parse failed, falling back to streaming parse")
        result = parse_check_response(review_lines)

    logger.info("[REVIEW RESULT] %s", json.dumps(result, ensure_ascii=False))

    if not result.get("reason"):
        assert False, (
            f"Eval {ctx.eval_id}: review result missing reason | "
            f"Review output: {json.dumps(result, ensure_ascii=False)}"
        )

    if result.get("status") != "pass":
        reason = result["reason"]
        msg = (
            f"Eval {ctx.eval_id}: expected_output check failed\n"
            f"Reviewer reason: {reason}\n"
            f"--- AI Response (by execution session) ---\n"
            f"{ctx.ai_text[:ctx.truncate_len]}\n"
            f"--- End AI Response ---"
        )
        assert False, msg


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

    skill_name = metafunc.config.getoption("--skill", None)
    eval_id = metafunc.config.getoption("--eval-id", None)

    test_cases: List[Dict[str, Any]] = []
    ids: List[str] = []

    for skill in get_skills_with_evals():
        if skill_name and skill != skill_name:
            continue
        evals_data = load_evals_md(skill)
        if not evals_data:
            continue
        skill_dir = get_skill_path(skill)
        for eval_item in evals_data.get("evals", []):
            if eval_id and str(eval_item.get("id")) != str(eval_id):
                continue
            test_cases.append(_build_eval_test_case(skill, skill_dir, eval_item))
            ids.append(f"{skill}::eval_{eval_item.get('id')}")

    metafunc.parametrize("eval_case", test_cases, ids=ids, scope="function")


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
    logger.debug("--- INPUT PROMPT ---")
    logger.debug(prompt)
    logger.debug("--- END INPUT PROMPT ---")
    if expected_output:
        logger.debug("--- EXPECTED OUTPUT ---")
        logger.debug(expected_output)
        logger.debug("--- END EXPECTED OUTPUT ---")


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


def _setup_eval_sandbox(sandbox_manager: SandboxManager, skill_name: str,
                       eval_id, skill_dir: Path,
                       distractor_skill_dirs: Optional[List[Path]] = None):
    """创建沙箱和 opencode runner"""
    sandbox_path = sandbox_manager.create_sandbox(skill_name, eval_id)
    sandbox_manager.create_skill_link(sandbox_path, skill_dir)
    for ds_dir in (distractor_skill_dirs or []):
        sandbox_manager.create_skill_link(sandbox_path, ds_dir)
    logs_dir = sandbox_manager.get_logs_dir(sandbox_path)
    opencode_runner = OpencodeRunner(
        keep_session=True,
        verbose=True,
        workdir=str(sandbox_path),
        session_dir=str(logs_dir)
    )
    return opencode_runner, sandbox_path


def _check_token_budget(eval_data: Dict[str, Any], eval_id, opencode_runner,
                        session_name: str) -> None:
    """检查 token 消耗是否超过硬性阈值"""
    max_tokens = eval_data.get("max_tokens")
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


def test_eval_case(eval_case: Dict[str, Any], sandbox_manager: SandboxManager):
    if os.environ.get("REPORT_ONLY") == "1":
        logger.info("[%s] REPORT_ONLY 模式，跳过测试执行 (eval %s)",
                    eval_case["skill_name"], eval_case["eval"].get("id"))
        return

    inputs = _unpack_eval_inputs(eval_case)

    opencode_runner, sandbox_path = _setup_eval_sandbox(
        sandbox_manager, inputs.skill_name, inputs.eval_id, inputs.skill_dir,
        distractor_skill_dirs=inputs.distractor_skill_dirs,
    )
    _log_eval_case_header(inputs.skill_name, inputs.eval_id, inputs.prompt,
                          inputs.expected_output,
                          distractor_skill_dirs=inputs.distractor_skill_dirs)

    session_name = f"{inputs.skill_name}_case_{inputs.eval_id}"
    full_output, error_output, session_file, success = _collect_exec_output(
        opencode_runner, inputs.prompt, ".", session_name,
    )
    assert success, f"Eval {inputs.eval_id}: opencode run failed - {error_output}"

    ai_text = extract_ai_text(full_output)
    logger.debug("--- AI Response (eval %s) ---", inputs.eval_id)
    logger.debug(ai_text[:1000])
    logger.debug("--- End AI Response ---")

    opencode_runner.export_session_data(
        output_file=str(opencode_runner.session_dir / f"{session_name}_ses.json")
    )
    _check_token_budget(inputs.eval_data, inputs.eval_id, opencode_runner, session_name)

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
    )
    validate_output(ctx)
    logger.info("Session file: %s", session_file)
