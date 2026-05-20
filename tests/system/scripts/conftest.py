# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import html as html_mod
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import pytest
import yaml

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

FRAMEWORK_DIR = Path(__file__).parent.parent  # skill-test-framework/
CONFIG_PATH = FRAMEWORK_DIR / "config" / "skill-test.config"
REPO_ROOT = FRAMEWORK_DIR.parent.parent  # 仓库根目录
EVALS_CASES_DIR = FRAMEWORK_DIR / "cases"  # 集中式 evals.json 存放目录
LOGS_DIR = FRAMEWORK_DIR / "logs"  # opencode session 导出 JSON 存放目录


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {"skill_dirs": ["skills"], "skill_whitelist": []}


CONFIG = load_config()


def get_skill_path(skill_name: str) -> Optional[Path]:
    """根据 skill 名称查找实际路径"""
    for skill_dir_rel in CONFIG.get("skill_dirs", ["skills"]):
        candidate = REPO_ROOT / skill_dir_rel / skill_name
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _patch_hydrate_data():
    """修复 pytest-html 的 _hydrate_data 方法，使其从含有 badge <span> 的
    结果单元格中正确提取纯文本（如 "Passed"）而非 HTML 片段。"""
    try:
        from pytest_html.basereport import BaseReport
    except ImportError:
        return

    original = getattr(BaseReport, '_hydrate_data', None)

    def patched_hydrate(self, data, cells):
        for index, cell in enumerate(cells):
            table_header = getattr(self, '_report', None)
            if table_header is None:
                continue
            if "sortable" in table_header.table_header[index]:
                name_match = re.search(r"col-(\w+)", cell)
                if not name_match:
                    continue
                col_name = name_match.group(1)
                # 去除 HTML 标签，只保留文本内容
                text = re.sub(r"<[^>]+>", "", cell).strip()
                data[col_name] = text

    setattr(BaseReport, '_hydrate_data', patched_hydrate)


def pytest_configure(config):
    # 强制 root logger 输出 DEBUG+ 级别的日志到 stderr。
    # pytest 的 LoggingPlugin 默认将 root logger 设为 WARNING 并添加 handler，
    # 导致其他模块中 logging.basicConfig(level=DEBUG) 成为空操作，所有
    # logger.info/debug() 被静默丢弃。此处绕过该限制，使日志消息能到达
    # report.capstderr，供 pytest_runtest_logreport 解析测试交互信息和评测得分。
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(handler)

    if sys.platform == 'win32':
        try:
            from pytest_html.html_report import HTMLReport
            original_save_report = getattr(HTMLReport, '_save_report', None)

            def patched_save_report(self, report_content):
                self.logfile.write_text(report_content, encoding='utf-8')

            setattr(HTMLReport, '_save_report', patched_save_report)
        except ImportError:
            pass

    # Monkey-patch _hydrate_data: 从单元格 HTML 中提取纯文本内容，
    # 兼容我们注入的 <span class="result-badge"> 标签。
    _patch_hydrate_data()


def pytest_addoption(parser):
    parser.addoption("--skill", action="store", default=None, help="Run evals for specific skill")
    parser.addoption("--eval-id", action="store", default=None, help="Run specific eval by ID")


def get_all_skills() -> List[str]:
    """
    扫描所有 skill_dirs 配置的目录，返回包含 SKILL.md 的 skill 名称列表。
    如果配置了 skill_whitelist，则只返回白名单中的 skill。
    """
    skills = set()
    skill_whitelist = CONFIG.get("skill_whitelist", [])
    for skill_dir_rel in CONFIG.get("skill_dirs", ["skills"]):
        skill_dir = REPO_ROOT / skill_dir_rel
        if not skill_dir.exists():
            continue
        for item in skill_dir.iterdir():
            if not item.is_dir():
                continue
            if skill_whitelist and item.name not in skill_whitelist:
                continue
            if (item / "SKILL.md").exists():
                skills.add(item.name)
    return sorted(skills)


def get_skills_with_evals() -> List[str]:
    """
    扫描 cases/ 目录，返回有 *_evals.json 文件的 skill 名称列表。
    如果配置了 skill_whitelist，则只返回白名单中的 skill。
    """
    skills = []
    skill_whitelist = CONFIG.get("skill_whitelist", [])
    if not EVALS_CASES_DIR.exists():
        return skills
    for f in EVALS_CASES_DIR.iterdir():
        if f.is_file() and f.name.endswith("_evals.json"):
            skill_name = f.name[:-len("_evals.json")]
            if skill_whitelist and skill_name not in skill_whitelist:
                continue
            skills.append(skill_name)
    return sorted(skills)


def load_evals_json(skill_name: str) -> Optional[Dict[str, Any]]:
    """从 cases/<skill_name>_evals.json 加载评测用例"""
    evals_path = EVALS_CASES_DIR / f"{skill_name}_evals.json"
    if not evals_path.exists():
        return None
    try:
        with open(evals_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


@pytest.fixture(scope="session")
def skills_dir() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def all_skills() -> List[str]:
    return get_all_skills()


@pytest.fixture(scope="session")
def skills_with_evals() -> List[str]:
    return get_skills_with_evals()


@pytest.fixture
def evals_data(request, skills_with_evals) -> Dict[str, Any]:
    skill_name = request.param
    data = load_evals_json(skill_name)
    if data is None:
        pytest.skip(f"No evals.json found for skill: {skill_name}")
    return data


@pytest.fixture
def skill_dir(request, skills_dir) -> Path:
    skill_name = request.param
    skill_path = get_skill_path(skill_name)
    if not skill_path:
        pytest.skip(f"Skill directory not found: {skill_name}")
    return skill_path


# ═══════════════════════════════════════════════════════════════
#  Custom CSS for HTML reports
# ═══════════════════════════════════════════════════════════════

REPORT_CSS = """
/* === Base === */
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 14px;
  color: #1e293b;
  background: #f8fafc;
  max-width: 1440px;
  margin: 0 auto;
  padding: 20px 24px;
}
h1 { font-size: 22px; color: #0f172a; margin: 0 0 4px; font-weight: 700; }
h2 { font-size: 16px; color: #334155; font-weight: 600; }

/* === Summary card === */
#environment {
  background: white;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  margin-bottom: 20px;
}
#environment td { padding: 8px 14px; }
#environment tr:first-child td { padding-top: 14px; }
#environment tr:last-child td { padding-bottom: 14px; }

/* === Filter bar === */
#filter-container { margin: 16px 0; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
#filter-container input[type="checkbox"] { accent-color: #6366f1; }
#filter-container label,
#filter-container span {
  font-size: 13px;
  padding: 4px 10px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.15s;
}
#filter-container label:hover { background: #e2e8f0; }
#filter-container .filter-header { font-weight: 600; color: #475569; cursor: default; }

/* === Result badges === */
.result-badge {
  display: inline-block;
  padding: 3px 12px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  line-height: 1.6;
}
.result-badge.passed  { background: #22c55e; color: #fff; }
.result-badge.failed  { background: #ef4444; color: #fff; }
.result-badge.skipped,
.result-badge.xfailed,
.result-badge.rerun   { background: #f59e0b; color: #fff; }
.result-badge.error,
.result-badge.xpassed { background: #ef4444; color: #fff; }

/* === Score badges === */
.score-badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 10px;
  font-size: 12px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}
.score-high { background: #dcfce7; color: #166534; }
.score-mid  { background: #fef3c7; color: #92400e; }
.score-low  { background: #fee2e2; color: #991b1b; }
.score-na   { color: #94a3b8; }

/* === Table === */
#results-table {
  border: none;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  width: 100%;
  font-size: 13px;
  background: white;
}
#results-table thead { background: #f1f5f9; }
#results-table th {
  padding: 10px 14px;
  text-align: left;
  font-weight: 600;
  color: #475569;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  border-bottom: 2px solid #e2e8f0;
  position: sticky;
  top: 0;
  z-index: 10;
}
#results-table th.sortable { cursor: pointer; }
#results-table th.sortable:hover { color: #1e293b; }
#results-table td {
  padding: 10px 14px;
  border-bottom: 1px solid #f1f5f9;
  vertical-align: top;
}
#results-table tbody tr:hover { background: #f1f5f9; }
#results-table tbody tr:nth-child(even) { background: #fafbfc; }
#results-table tbody tr:nth-child(even):hover { background: #f1f5f9; }
.col-result { width: 120px; text-align: center; }
.col-skill  { width: 160px; font-weight: 500; color: #334155; }
.col-description { width: 220px; color: #475569; font-size: 13px; }
.col-score { width: 90px; text-align: center; }
.col-testId { width: auto; font-family: "JetBrains Mono", "Fira Code", monospace; font-size: 12px; }
.col-duration { width: 90px; text-align: right; color: #94a3b8; font-variant-numeric: tabular-nums; }
.col-links  { width: 40px; text-align: center; }

/* === Collapse/expand === */
.col-result.collapsed { cursor: pointer; }
.col-result.collapsed::after {
  content: " \\25B6";
  font-size: 9px;
  margin-left: 4px;
  color: #94a3b8;
}
.col-result:not(.collapsed) { cursor: pointer; }
.col-result:not(.collapsed)::after {
  content: " \\25BC";
  font-size: 9px;
  margin-left: 4px;
  color: #94a3b8;
}
.extras-row.hidden { display: none; }

/* === Log area (terminal style) === */
.logwrapper { margin-top: 8px; }
.logexpander {
  cursor: pointer;
  padding: 6px 14px;
  background: #f1f5f9;
  border-radius: 6px 6px 0 0;
  font-size: 12px;
  color: #64748b;
  user-select: none;
  border: 1px solid #e2e8f0;
  border-bottom: none;
}
.logexpander:hover { background: #e2e8f0; }
.logexpander::after { content: " \\25BC \\65E5\\5FD7"; }
.logwrapper:not(.expanded) .logexpander::after { content: " \\25B6 \\65E5\\5FD7"; }
.log {
  background: #f8fafc;
  color: #334155;
  padding: 14px;
  border-radius: 0 0 6px 6px;
  font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", monospace;
  font-size: 12px;
  line-height: 1.7;
  max-height: 500px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-word;
  border: 1px solid #e2e8f0;
  border-top: none;
}
.logwrapper:not(.expanded) .log { display: none; }
.log .error { color: #dc2626; font-weight: 600; }

/* === Failure detail blocks === */
.extra { padding: 12px !important; background: #fafbfc; }

.failure-block {
  margin: 10px 0;
  padding: 14px 18px;
  border-radius: 8px;
  border-left: 4px solid #94a3b8;
  background: white;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.failure-block + .failure-block { margin-top: 12px; }

.failure-label {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  margin-bottom: 8px;
}
.failure-content {
  font-size: 14px;
  line-height: 1.65;
  color: #1e293b;
}
.failure-content code {
  background: #f1f5f9;
  padding: 1px 6px;
  border-radius: 3px;
  font-family: "JetBrains Mono", monospace;
  font-size: 12px;
}
.failure-code {
  font-family: "JetBrains Mono", "Fira Code", monospace;
  font-size: 12px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 420px;
  overflow-y: auto;
  background: #f8fafc;
  padding: 12px;
  border-radius: 6px;
  margin: 0;
  color: #334155;
  border: 1px solid #e2e8f0;
}

.failure-reviewer-reason { border-left-color: #f97316; background: #fff7ed; }
.failure-reviewer-reason .failure-label { color: #c2410c; }

.failure-ai-response { border-left-color: #6366f1; background: #eef2ff; }
.failure-ai-response .failure-label { color: #4338ca; }

.failure-pattern { border-left-color: #eab308; background: #fefce8; }
.failure-pattern .failure-label { color: #a16207; }

.failure-error { border-left-color: #ef4444; background: #fef2f2; }
.failure-error .failure-label { color: #dc2626; }

/* === Phase 2 structured log blocks === */
.log-block {
  margin: 10px 0;
  padding: 12px 16px;
  border-radius: 8px;
  border-left: 4px solid #94a3b8;
  background: #fff;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.log-block-label {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
}
.log-block-content {
  font-size: 13px;
  line-height: 1.6;
  color: #334155;
  white-space: pre-wrap;
  word-break: break-word;
}
.log-block-content.log-block-code {
  font-family: "JetBrains Mono", "Fira Code", monospace;
  font-size: 12px;
  max-height: 360px;
  overflow-y: auto;
  background: #f8fafc;
  padding: 10px;
  border-radius: 4px;
  border: 1px solid #e2e8f0;
}

.log-input-prompt .log-block-label    { color: #4338ca; }
.log-input-prompt    { border-left-color: #6366f1; background: #eef2ff; }
.log-expected-output .log-block-label { color: #15803d; }
.log-expected-output { border-left-color: #22c55e; background: #f0fdf4; }
.log-ai-reasoning .log-block-label    { color: #7e22ce; }
.log-ai-reasoning    { border-left-color: #a855f7; background: #faf5ff; }
.log-ai-response .log-block-label     { color: #475569; }
.log-ai-response     { border-left-color: #64748b; background: #f8fafc; }
.log-review-pass .log-block-label     { color: #15803d; }
.log-review-pass     { border-left-color: #22c55e; background: #f0fdf4; }
.log-review-fail .log-block-label     { color: #dc2626; }
.log-review-fail     { border-left-color: #ef4444; background: #fef2f2; }
.log-review-prompt .log-block-label   { color: #0f766e; }
.log-review-prompt   { border-left-color: #14b8a6; background: #f0fdfa; }

/* === Environment toggle === */
#environment-header h2 { cursor: pointer; }
#environment-header.collapsed h2::after { content: " \\25B6"; font-size: 12px; }
#environment-header:not(.collapsed) h2::after { content: " \\25BC"; font-size: 12px; }

/* === Responsive === */
@media (max-width: 768px) {
  body { padding: 10px; font-size: 13px; }
  #results-table { font-size: 12px; }
  #results-table th, #results-table td { padding: 6px 8px; }
  .col-skill { width: 100px; }
}
"""


# ═══════════════════════════════════════════════════════════════
#  Helpers for hooks
# ═══════════════════════════════════════════════════════════════

def _extract_skill_name(nodeid: str) -> str:
    """从 pytest nodeid 提取 skill 名称"""
    matches = re.findall(r'\[(.*?)\]', nodeid)
    if matches:
        param = matches[-1]
        if '::' in param:
            return param.split('::')[0]
        return param
    return "—"


TEST_DESCRIPTIONS = {
    # TestEvalsJsonStructure
    "test_evals_json_exists": "evals.json 文件存在性",
    "test_evals_json_valid": "evals.json JSON 格式合法性",
    "test_evals_json_has_skill_name": "evals.json 包含 skill_name 字段",
    "test_evals_json_has_evals_list": "evals.json 包含 evals 列表",
    # TestEvalCaseStructure
    "test_eval_cases_have_id": "评测用例具有 id 字段",
    "test_eval_cases_have_prompt": "评测用例具有 prompt 字段",
    "test_eval_cases_have_expected_output": "评测用例具有 expected_output 字段",
    "test_eval_cases_have_files": "评测用例具有 files 字段",
    "test_eval_cases_expectations_format": "expectations 字段格式合法",
    # TestEvalCaseLogic
    "test_eval_ids_are_unique": "用例 ID 唯一性",
    "test_eval_ids_are_sequential": "用例 ID 连续递增",
    "test_prompt_is_descriptive": "prompt 非空（描述性检查）",
    "test_expected_output_matches_prompt": "expected_output 长度检查",
    # TestSkillDirectory
    "test_skill_has_skill_md": "SKILL.md 文件存在性",
    "test_skill_md_has_frontmatter": "SKILL.md YAML frontmatter 格式",
    "test_skill_md_has_required_fields": "SKILL.md frontmatter 必填字段",
    # Phase 2: test_skill_evals.py
    "test_eval_case": "AI 语义评测",
}


def _get_test_description(nodeid: str) -> str:
    """从 nodeid 提取测试函数名，返回中文描述"""
    m = re.search(r'::(\w+)(?:\[|$)', nodeid)
    if m:
        func_name = m.group(1)
        return TEST_DESCRIPTIONS.get(func_name, func_name.replace("_", " "))
    return nodeid


def _parse_reviewer_reason_block(longrepr: str, eval_id: str) -> Optional[str]:
    """解析 reviewer reason 失败块"""
    if "expected_output check failed" not in longrepr:
        return None
    reason_match = re.search(
        r'Reviewer reason:\s*(.+?)(?:\n---|\nassert|\Z)',
        longrepr, re.DOTALL
    )
    reason = html_mod.escape(reason_match.group(1).strip()) if reason_match else "unknown"
    return (
        f'<div class="failure-block failure-reviewer-reason">\n'
        f'  <div class="failure-label">✔ 评测判定 — Eval {eval_id}</div>\n'
        f'  <div class="failure-content">{reason}</div>\n'
        f'</div>'
    )


def _parse_pattern_block(longrepr: str, eval_id: str) -> Optional[str]:
    """解析模式匹配失败块"""
    if "expected pattern not found:" in longrepr:
        pm = re.search(r"expected pattern not found:\s*'(.+?)'", longrepr)
        pattern = html_mod.escape(pm.group(1)) if pm else "?"
        return (
            f'<div class="failure-block failure-pattern">\n'
            f'  <div class="failure-label">✖ 模式匹配失败 — Eval {eval_id}</div>\n'
            f'  <div class="failure-content">'
            f'输出中<strong>未找到</strong>期望的模式: '
            f'<code>{pattern}</code></div>\n</div>'
        )
    if "unexpected pattern found:" in longrepr:
        pm = re.search(r"unexpected pattern found:\s*'(.+?)'", longrepr)
        pattern = html_mod.escape(pm.group(1)) if pm else "?"
        return (
            f'<div class="failure-block failure-pattern">\n'
            f'  <div class="failure-label">✖ 意外模式匹配 — Eval {eval_id}</div>\n'
            f'  <div class="failure-content">'
            f'输出中<strong>不应包含</strong>: '
            f'<code>{pattern}</code></div>\n</div>'
        )
    return None


def _parse_execution_error_block(longrepr: str, eval_id: str) -> Optional[str]:
    """解析执行错误块"""
    if "opencode run failed" not in longrepr and "review session error" not in longrepr:
        return None
    msg_match = re.search(
        r'(?:opencode run failed|review session error)\s*[-:]\s*(.+?)(?:\n---|\nassert|\Z)',
        longrepr, re.DOTALL
    )
    msg = html_mod.escape(msg_match.group(1).strip()) if msg_match else "unknown error"
    return (
        f'<div class="failure-block failure-error">\n'
        f'  <div class="failure-label">✖ 执行错误 — Eval {eval_id}</div>\n'
        f'  <div class="failure-content">{msg}</div>\n'
        f'</div>'
    )


def _parse_ai_response_block(longrepr: str) -> Optional[str]:
    """提取 AI 回复原文块"""
    ai_match = re.search(
        r'(?:^|\n)\s*--- AI Response[^\n]*\n(.*?)\n\s*--- End AI Response',
        longrepr, re.DOTALL
    )
    if not ai_match:
        return None
    ai_text = html_mod.escape(ai_match.group(1).strip())
    return (
        f'<div class="failure-block failure-ai-response">\n'
        f'  <div class="failure-label">\U0001F4AC AI 回复原文</div>\n'
        f'  <pre class="failure-code">{ai_text}</pre>\n'
        f'</div>'
    )


def _parse_fallback_error_block(longrepr: str) -> str:
    """兜底：无结构化标记时展示关键错误行"""
    lines = longrepr.strip().split('\n')
    msg_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('E   '):
            clean = stripped[4:].strip()
            if clean and not clean.startswith(' ' * 20):
                msg_lines.append(html_mod.escape(clean))
        elif 'AssertionError' in stripped:
            msg_lines.append(html_mod.escape(stripped))
        elif 'assert' in stripped.lower() and len(stripped) < 200:
            msg_lines.append(html_mod.escape(stripped))

    error_text = '\n'.join(msg_lines[:15]) if msg_lines else html_mod.escape(longrepr[:800])
    return (
        f'<div class="failure-block failure-error">\n'
        f'  <div class="failure-label">✖ 错误详情</div>\n'
        f'  <pre class="failure-code">{error_text}</pre>\n'
        f'</div>'
    )


def _parse_failure_to_html(longrepr: str) -> str:
    """解析断言失败文本，提取结构化信息生成 HTML"""
    if not longrepr:
        return ""

    blocks = []

    eval_id = "?"
    m = re.search(r'Eval (\d+)[,:]', longrepr)
    if m:
        eval_id = m.group(1)

    for parser in (
        _parse_reviewer_reason_block,
        _parse_pattern_block,
        _parse_execution_error_block,
    ):
        block = parser(longrepr, eval_id)
        if block:
            blocks.append(block)
            break

    ai_block = _parse_ai_response_block(longrepr)
    if ai_block:
        blocks.append(ai_block)

    if not blocks:
        blocks.append(_parse_fallback_error_block(longrepr))

    return '\n'.join(blocks)


def strip_markdown_fence(text: str) -> str:
    """去除 markdown 代码块包裹 ```json ... ```"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    return cleaned


def get_opencode_text(data: Dict[str, Any]) -> str:
    """从 opencode JSON 事件中提取文本内容"""
    return data.get("part", {}).get("text", "") or data.get("text", "")


def _build_log_block(label: str, content: str, css_class: str, is_code: bool = False) -> str:
    """构建结构化日志 HTML 卡片"""
    safe_content = html_mod.escape(content.strip())
    content_cls = "log-block-content log-block-code" if is_code else "log-block-content"
    return (
        f'<div class="log-block {css_class}">\n'
        f'  <div class="log-block-label">{label}</div>\n'
        f'  <div class="log-block-content {content_cls}">{safe_content}</div>\n'
        f'</div>'
    )


def extract_review_json(text: str) -> Optional[Dict[str, Any]]:
    """从文本中提取评测结果 JSON，兼容 markdown 代码块和裸 JSON"""
    # 策略1: ```json ... ```
    for m in re.finditer(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL):
        try:
            result = json.loads(m.group(1).strip())
            if result.get("status") in ("pass", "fail"):
                return result
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    # 策略2: 裸 JSON 对象含 "status": "pass"/"fail"
    for m in re.finditer(r'\{[^{}]*"status"\s*:\s*"(?:pass|fail)"[^{}]*\}', text, re.DOTALL):
        try:
            result = json.loads(m.group())
            if result.get("status") in ("pass", "fail"):
                return result
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    # 策略3: 去 markdown 围栏后解析全文
    cleaned = strip_markdown_fence(text)
    if cleaned != text:
        try:
            result = json.loads(cleaned)
            if result.get("status") in ("pass", "fail"):
                return result
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return None


def _get_text_from_parts(parts: List[Dict]) -> str:
    """从 opencode message parts 中提取所有文本内容"""
    texts = []
    for p in parts:
        if p.get("type") in ("text", "reasoning"):
            t = p.get("text", "") or p.get("part", {}).get("text", "")
            if t.strip():
                texts.append(t)
    return "\n".join(texts)


def _load_json_file(file_path: Path) -> Dict[str, Any]:
    """安全加载 JSON 文件"""
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _extract_expected_points_block(review_messages: List[Dict]) -> Optional[str]:
    """从评测消息中提取预期要点块"""
    if not review_messages:
        return None
    user_parts = review_messages[0].get("parts", [])
    review_prompt_text = _get_text_from_parts(user_parts)
    m = re.search(
        r'###\s+预期回复应覆盖的要点\s*\n(.*?)(?:\n###|\Z)',
        review_prompt_text, re.DOTALL
    )
    if m and m.group(1).strip():
        return _build_log_block("预期要点", m.group(1), "log-expected-output")
    return None


def _extract_review_result_block(part: Dict) -> Optional[tuple]:
    """从单个 part 中提取评测结果，返回 (block, score) 或 None"""
    if part.get("type") != "text":
        return None
    text = get_opencode_text(part)
    result = extract_review_json(text)
    if not result:
        return None
    status = result.get("status", "fail")
    score = result.get("score")
    reason = result.get("reason", "")
    cls = "log-review-pass" if status == "pass" else "log-review-fail"
    label = "评测通过" if status == "pass" else "评测未通过"
    if score is not None:
        label += f" ({score}/100)"
    block = _build_log_block(label, reason if reason else status, cls)
    return block, score


def _extract_review_blocks(review_messages: List[Dict]) -> tuple:
    """从评测 session 消息中提取预期要点和评测结果，返回 (blocks, score)"""
    blocks = []
    score = None

    expected_block = _extract_expected_points_block(review_messages)
    if expected_block:
        blocks.append(expected_block)

    for msg in review_messages[1:]:
        for part in msg.get("parts", []):
            result = _extract_review_result_block(part)
            if result:
                blocks.append(result[0])
                return blocks, result[1]

    return blocks, score


def _extract_prompt_block(ses_messages: List[Dict]) -> Optional[str]:
    """从执行消息中提取输入 Prompt 块"""
    if not ses_messages:
        return None
    user_parts = ses_messages[0].get("parts", [])
    prompt_text = _get_text_from_parts(user_parts)
    if prompt_text.strip():
        return _build_log_block("输入 Prompt", prompt_text, "log-input-prompt")
    return None


def _collect_reasoning_and_response_texts(ses_messages: List[Dict]) -> tuple:
    """从执行消息中收集思考过程和回复文本，返回 (reasoning_texts, response_texts)"""
    reasoning_texts = []
    response_texts = []
    for msg in ses_messages[1:]:
        for part in msg.get("parts", []):
            ptype = part.get("type")
            t = get_opencode_text(part)
            if not t.strip():
                continue
            if ptype == "reasoning":
                reasoning_texts.append(t)
            elif ptype == "text":
                response_texts.append(t)
    return reasoning_texts, response_texts


def _extract_session_blocks(ses_messages: List[Dict]) -> List[str]:
    """从执行 session 消息中提取 Prompt、思考过程和回复"""
    blocks = []

    prompt_block = _extract_prompt_block(ses_messages)
    if prompt_block:
        blocks.append(prompt_block)

    if len(ses_messages) <= 1:
        return blocks

    reasoning_texts, response_texts = _collect_reasoning_and_response_texts(ses_messages)

    if reasoning_texts:
        blocks.append(_build_log_block(
            "AI 思考过程", "\n".join(reasoning_texts), "log-ai-reasoning"
        ))
    if response_texts:
        blocks.append(_build_log_block(
            "AI 回复", "\n".join(response_texts),
            "log-ai-response", is_code=True
        ))

    return blocks


def _build_phase2_html_from_json(skill_name: str, eval_id):
    """从 logs 目录下的 JSON 文件解析测试交互信息，生成 HTML 卡片。
    返回 (html: str, score: int | None) 元组。"""
    ses_file = LOGS_DIR / f"{skill_name}_case_{eval_id}_ses.json"
    review_file = LOGS_DIR / f"{skill_name}_case_{eval_id}_review_ses.json"

    if not review_file.exists():
        return "", None

    review_data = _load_json_file(review_file)
    review_messages = review_data.get("messages", [])

    review_blocks, score = _extract_review_blocks(review_messages)

    ses_data = _load_json_file(ses_file)
    ses_messages = ses_data.get("messages", [])
    session_blocks = _extract_session_blocks(ses_messages)

    blocks = session_blocks[:1] + review_blocks + session_blocks[1:]
    return '\n'.join(blocks), score


# ═══════════════════════════════════════════════════════════════
#  pytest-html hooks
# ═══════════════════════════════════════════════════════════════

def pytest_html_report_title(report):
    report.title = "Skill Test Report"


def pytest_html_results_summary(prefix, summary, postfix, session):
    prefix.append(f"<style>{REPORT_CSS}</style>")


def pytest_html_results_table_header(cells):
    cells.insert(1, '<th class="sortable" data-column-type="skill">Skill</th>')
    cells.insert(2, '<th>描述</th>')
    cells.insert(3, '<th>评测得分</th>')


def pytest_html_results_table_row(report, cells):
    # 结果徽章 — 先提取 cells[0] 中的原始结果文本
    m = re.search(r'>([^<]+)<', cells[0])
    if m:
        result_text = m.group(1).strip()
        result_class = result_text.lower()
        cells[0] = (
            f'<td class="col-result">'
            f'<span class="result-badge {result_class}">'
            f'{html_mod.escape(result_text)}</span></td>'
        )

    # Skill 列 — 插入在 Result 之后、Test 之前
    skill_name = _extract_skill_name(report.nodeid)
    cells.insert(1, f'<td class="col-skill">{html_mod.escape(skill_name)}</td>')

    # 描述列 — 插入在 Skill 之后、Test 之前
    desc = _get_test_description(report.nodeid)
    cells.insert(2, f'<td class="col-description">{html_mod.escape(desc)}</td>')

    # 评测得分列 — 插入在描述之后、Test 之前
    score = getattr(report, '_eval_score', None)
    if score is None:
        # Fallback for xdist: _eval_score is a custom attr not serialized
        # between worker and master, but user_properties IS transported.
        for key, val in getattr(report, 'user_properties', []) or []:
            if key == 'eval_score':
                score = val
                break
    if score is not None:
        if score >= 80:
            score_cls = "score-high"
        elif score >= 60:
            score_cls = "score-mid"
        else:
            score_cls = "score-low"
        score_html = f'<span class="score-badge {score_cls}">{score}</span>'
    else:
        score_html = f'<span class="score-badge score-na">&mdash;</span>'
    cells.insert(3, f'<td class="col-score">{score_html}</td>')


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    """注入结构化的 extra HTML：失败用例解析断言信息，Phase 2 用例从 logs/ JSON 文件解析"""
    if report.when != "call":
        return

    from pytest_html import extras

    extra_items = list(getattr(report, 'extras', []))

    if report.failed:
        failure_html = _parse_failure_to_html(
            getattr(report, 'longreprtext', '') or ''
        )
        if failure_html:
            extra_items.append(extras.html(failure_html))

    # 从 nodeid 解析 skill_name / eval_id，读取 logs/ 目录下的 JSON 文件
    skill_name = _extract_skill_name(report.nodeid)
    eval_id = None
    # nodeid 格式: test_skill_evals.py::test_eval_case[skill::eval_N]
    # 文件名格式: skill_case_N_review_ses.json
    bracket_match = re.search(r'\[.*?::eval_(\d+)\]', report.nodeid)
    if bracket_match:
        eval_id = bracket_match.group(1)

    if skill_name and eval_id:
        # 去重：xdist 并行时 worker 和 master 各触发一次此钩子。
        # master 侧 extras 已反序列化为 dict，需同时兼容 dict 和 Extra 对象。
        has_phase2 = any(
            'log-block' in (item.get('content', '') if isinstance(item, dict) else getattr(item, 'content', ''))
            for item in extra_items
        )
        phase2_html, score = _build_phase2_html_from_json(skill_name, eval_id)

        if phase2_html and not has_phase2:
            extra_items.append(extras.html(phase2_html))

        if score is not None and getattr(report, '_eval_score', None) is None:
            setattr(report, '_eval_score', score)
            if not hasattr(report, 'user_properties'):
                report.user_properties = []
            report.user_properties.append(("eval_score", score))

    if extra_items:
        report.extras = extra_items
