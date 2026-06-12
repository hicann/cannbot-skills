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
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

import pytest
import yaml

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

FRAMEWORK_DIR = Path(__file__).parent.parent  # skill-test-framework/
CONFIG_PATH = FRAMEWORK_DIR / "config" / "st-test.config"
REPO_ROOT = FRAMEWORK_DIR.parent.parent  # 仓库根目录
EVALS_CASES_DIR = FRAMEWORK_DIR / "cases"  # 集中式 evals 存放目录
LOGS_DIR = FRAMEWORK_DIR / "logs"  # opencode session 导出 JSON 存放目录
SANDBOX_DIR = FRAMEWORK_DIR / "sandboxes"  # 沙箱隔离目录

# ── 评测维度阈值常量 ──────────────────────────────────────────────
# AI 评审模型在 dimensions 字段中以结构化格式报告各维度得分
DEFAULT_DIMENSION_THRESHOLDS: Dict[str, int] = {
    "覆盖度": 20,   # max 40
    "准确性": 15,   # max 30
    "质量": 10,     # max 20
    "Token": 3,     # max 10
}

DIMENSION_MAX_SCORES: Dict[str, int] = {
    "覆盖度": 40,
    "准确性": 30,
    "质量": 20,
    "Token": 10,
}

DIMENSION_ORDER: List[str] = ["覆盖度", "准确性", "质量", "Token"]

# 维度名归一化映射：AI 可能输出的各种变体，统一映射为标准名
DIMENSION_NAME_NORMALIZE = {
    "覆盖度": "覆盖度",
    "准确性": "准确性",
    "技术准确性": "准确性",
    "质量": "质量",
    "回复质量": "质量",
    "Token": "Token",
    "Token消耗": "Token",
    "Token 消耗": "Token",
    "token消耗": "Token",
    "token 消耗": "Token",
    "token": "Token",
}


def parse_dimension_scores(dimensions: Optional[Dict] = None) -> Dict[str, int]:
    """从 AI 评审结果 dimensions 字段解析各维度得分。

    Input: {"覆盖度": {"score": 38, "max": 40}, "准确性": {"score": 27, "max": 30}, ...}
    Output: {"覆盖度": 38, "准确性": 27, "质量": 15, "Token": 8}
    """
    if not dimensions:
        return {}
    scores: Dict[str, int] = {}
    for dim_name, dim_data in dimensions.items():
        normalized = DIMENSION_NAME_NORMALIZE.get(dim_name, dim_name)
        if isinstance(dim_data, dict) and "score" in dim_data:
            try:
                scores[normalized] = int(dim_data["score"])
            except (ValueError, TypeError):
                continue
    return scores


def validate_dimension_scores(
    dim_scores: Dict[str, int],
    thresholds: Dict[str, int],
    eval_id: str = "",
) -> Optional[str]:
    """校验各维度分数是否达到阈值，返回错误信息或 None。"""
    failures: List[str] = []
    for dim, threshold in thresholds.items():
        score = dim_scores.get(dim)
        if score is None:
            failures.append(f"{dim}: score not found in reason field")
        elif score < threshold:
            max_score = DIMENSION_MAX_SCORES.get(dim, "?")
            failures.append(f"{dim} ({score}/{max_score}) 低于阈值 ({threshold})")
    if failures:
        tag = f"(Eval {eval_id}) " if eval_id else ""
        return tag + "维度阈值检查不通过: " + "; ".join(failures)
    return None


def parse_review_md(content: str) -> Dict[str, Any]:
    """从填写完成的评审模板 MD 文档中提取结构化评审结果。

    解析 review-template.md 被评审 Agent 填写后的内容，
    提取 Status、Total Score、各维度得分和 Review Comments。
    返回与旧 JSON 格式兼容的 dict，可直接传给 parse_dimension_scores()
    和 validate_dimension_scores() 进行程序化阈值校验。

    Args:
        content: 评审 Agent 填写后的 MD 模板全文。

    Returns:
        {
            "status": "pass" | "fail" | "error",
            "score": 85,
            "dimensions": {
                "覆盖度": {"score": 38, "max": 40},
                "准确性": {"score": 27, "max": 30},
                "质量": {"score": 15, "max": 20},
                "Token": {"score": 8, "max": 10},
            },
            "reason": "详细评审意见文本...",
        }
    """
    def _find(pattern, text, default=None, flags=0):
        m = re.search(pattern, text, flags)
        return m.group(1).strip() if m else default

    # 1. 提取 Status
    status_raw = _find(r'-\s*Status:\s*(PASS|FAIL)', content, flags=re.IGNORECASE)
    status = status_raw.lower() if status_raw else "fail"

    # 2. 提取 Total Score
    score_raw = _find(r'-\s*Total\s*Score:\s*(\d+)', content, flags=re.IGNORECASE)
    try:
        score = int(score_raw) if score_raw else 0
    except (ValueError, TypeError):
        score = 0

    # 3. 提取维度表格：匹配 | 维度名 | 得分 | 满分 | 阈值 | YES/NO |
    dimensions = {}
    _dim_table_re = re.compile(
        r'\|\s*(覆盖度|准确性|质量|Token)\s*\|'
        r'\s*(\d+)\s*\|'
        r'\s*(\d+)\s*\|'
        r'\s*\d+\s*\|'
        r'\s*(YES|NO)\s*\|'
    )
    for row_match in _dim_table_re.finditer(content):
        dim_name = row_match.group(1)
        dim_score = int(row_match.group(2))
        dim_max = int(row_match.group(3))
        dimensions[dim_name] = {"score": dim_score, "max": dim_max}

    # 补全缺失维度
    for dim in DIMENSION_ORDER:
        if dim not in dimensions:
            dimensions[dim] = {"score": 0, "max": DIMENSION_MAX_SCORES.get(dim, 0)}

    # 4. 提取 Review Comments（从 "## Review Comments" 到下一个 "## " 或 EOF）
    reason = _find(r'## Review Comments\s*\n+(.*?)(?=\n## |\Z)', content, "", re.DOTALL)
    reason = (reason or "").strip()
    if not reason:
        # Fallback: 无 lookahead 兼容
        m = re.search(r'## Review Comments\s*\n+(.*)', content, re.DOTALL)
        if m:
            reason = m.group(1).strip()

    # 5. 容错：若占位符完全未被填充，视为解析失败
    if score == 0 and status == "fail" and not reason:
        status = "error"
        reason = "无法从模板文件中解析评审结果（占位符可能未被填充）"

    return {
        "status": status,
        "score": score,
        "dimensions": dimensions,
        "reason": reason or "(no review comments found)",
    }


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


def _patch_write_report():
    """Monkey-patch BaseReport._write_report 以注入文档指引 footer。

    同时修复 Windows 上编码问题（原补丁针对不存在的 HTMLReport._save_report 无效）。
    """
    try:
        from pytest_html.basereport import BaseReport
    except ImportError:
        return

    original_write_report = getattr(BaseReport, '_write_report')

    def patched_write_report(self, rendered_report):
        footer_html = _build_footer_html()
        rendered_report = rendered_report.replace("</body>", footer_html + "\n</body>", 1)
        original_write_report(self, rendered_report)

    setattr(BaseReport, '_write_report', patched_write_report)


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

    # Monkey-patch BaseReport._write_report：
    # 1. 在报告底部注入 ST 用例开发指引 footer
    # 2. 修复 Windows 编码问题（原 HTMLReport._save_report 在 pytest-html 4.2.0 中不存在）
    _patch_write_report()

    # Monkey-patch _hydrate_data: 从单元格 HTML 中提取纯文本内容，
    # 兼容我们注入的 <span class="result-badge"> 标签。
    _patch_hydrate_data()


def pytest_addoption(parser):
    parser.addoption("--skill", action="append", default=None, help="Run evals for specific skill(s)")
    parser.addoption("--team", action="append", default=None, help="Run evals for specific team(s)")
    parser.addoption("--eval-id", action="store", default=None, help="Run specific eval by ID")
    parser.addoption("--ascend-platform", action="append", default=None,
                     help="Filter eval cases by Ascend platform (A2/A3/A5). Repeatable.")


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
    扫描 cases/ 目录，返回有 *_evals.md 文件的 skill 名称列表。
    如果配置了 skill_whitelist，则只返回白名单中的 skill。
    """
    skills = []
    skill_whitelist = CONFIG.get("skill_whitelist", [])
    if not EVALS_CASES_DIR.exists():
        return skills
    for f in EVALS_CASES_DIR.iterdir():
        if f.is_file() and f.name.endswith("_evals.md"):
            skill_name = f.name[:-len("_evals.md")]
            if skill_whitelist and skill_name not in skill_whitelist:
                continue
            skills.append(skill_name)
    return sorted(skills)


def load_evals_md(skill_name: str) -> Optional[Dict[str, Any]]:
    """从 cases/<skill_name>_evals.md 加载评测用例"""
    from evals_parser import parse_evals_md

    evals_path = EVALS_CASES_DIR / f"{skill_name}_evals.md"
    return parse_evals_md(evals_path)


# ── Team 发现函数 ──────────────────────────────────────────────────

def get_team_path(team_name: str) -> Optional[Path]:
    """根据 team 名称查找实际路径"""
    for team_dir_rel in CONFIG.get("team_dirs", []):
        candidate = REPO_ROOT / team_dir_rel / team_name
        if candidate.exists() and candidate.is_dir():
            plugin_json = candidate / ".claude-plugin" / "plugin.json"
            agents_md = candidate / "AGENTS.md"
            if plugin_json.exists() and agents_md.exists():
                return candidate
    return None


def get_all_teams() -> List[str]:
    """扫描所有 team_dirs 配置的目录，返回有 AGENTS.md + plugin.json 的 team 名称列表。"""
    teams = set()
    team_whitelist = CONFIG.get("team_whitelist", [])
    for team_dir_rel in CONFIG.get("team_dirs", []):
        team_dir = REPO_ROOT / team_dir_rel
        if not team_dir.exists():
            continue
        for item in team_dir.iterdir():
            if not item.is_dir():
                continue
            if team_whitelist and item.name not in team_whitelist:
                continue
            plugin_json = item / ".claude-plugin" / "plugin.json"
            agents_md = item / "AGENTS.md"
            if plugin_json.exists() and agents_md.exists():
                teams.add(item.name)
    return sorted(teams)


def get_teams_with_evals() -> List[str]:
    """扫描 cases/ 目录，返回 team_name 匹配且在白名单中的 team 名列表。"""
    teams = []
    team_whitelist = CONFIG.get("team_whitelist", [])
    if not EVALS_CASES_DIR.exists():
        return teams
    from evals_parser import parse_evals_md

    for f in EVALS_CASES_DIR.iterdir():
        if not f.is_file() or not f.name.endswith("_evals.md"):
            continue
        try:
            data = parse_evals_md(f)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", f.name, e)
            continue
        if not data or data.get("target_type") != "team":
            continue
        team_name = data.get("team_name", "")
        if not team_name:
            continue
        if team_whitelist and team_name not in team_whitelist:
            continue
        teams.append(team_name)
    return sorted(teams)


def load_team_evals_md(team_name: str) -> Optional[Dict[str, Any]]:
    """从 cases/<team_name>_evals.md 加载 team 评测用例"""
    from evals_parser import parse_evals_md

    evals_path = EVALS_CASES_DIR / f"{team_name}_evals.md"
    return parse_evals_md(evals_path)


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
    data = load_evals_md(skill_name)
    if data is None:
        pytest.skip(f"No evals.md found for skill: {skill_name}")
    return data


@pytest.fixture
def skill_dir(request, skills_dir) -> Path:
    skill_name = request.param
    skill_path = get_skill_path(skill_name)
    if not skill_path:
        pytest.skip(f"Skill directory not found: {skill_name}")
    return skill_path


# ── Team fixtures ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def all_teams() -> List[str]:
    return get_all_teams()


@pytest.fixture(scope="session")
def teams_with_evals() -> List[str]:
    return get_teams_with_evals()


@pytest.fixture
def team_evals_data(request, teams_with_evals) -> Dict[str, Any]:
    team_name = request.param
    data = load_team_evals_md(team_name)
    if data is None:
        pytest.skip(f"No evals.md found for team: {team_name}")
    return data


@pytest.fixture
def team_dir(request, skills_dir) -> Path:
    team_name = request.param
    team_path = get_team_path(team_name)
    if not team_path:
        pytest.skip(f"Team directory not found: {team_name}")
    return team_path


# ── Shared helpers ────────────────────────────────────────────────


def create_opencode_runner(sandbox_manager, sandbox_path, timeout=None):
    """创建 OpencodeRunner 实例（skill/team 共用）。

    消除 test_skill_evals.py 与 test_team_evals.py 中重复的
    OpencodeRunner 构造代码。
    """
    from opencode_runner import OpencodeRunner

    logs_dir = sandbox_manager.get_logs_dir(sandbox_path)
    model = os.environ.get("EVAL_MODEL")
    variant = os.environ.get("EVAL_MODEL_VARIANT")
    return OpencodeRunner(
        model=model,
        variant=variant,
        keep_session=True,
        verbose=True,
        workdir=str(sandbox_path),
        session_dir=str(logs_dir),
        timeout=timeout if timeout is not None else 600,
    )


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
.score-badge[title] { cursor: help; border-bottom: 1px dotted currentColor; }

/* === Rating badges === */
.rating-badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 10px;
  font-size: 12px;
  font-weight: 700;
}
.rating-s { background: #166534; color: #fff; }
.rating-a { background: #dcfce7; color: #166534; }
.rating-b { background: #dbeafe; color: #1e40af; }
.rating-c { background: #fef3c7; color: #92400e; }
.rating-d { background: #fee2e2; color: #991b1b; }
.rating-na { color: #94a3b8; }

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
.col-type  { width: 80px; text-align: center; font-weight: 600; color: #475569; }
.col-skill  { width: 160px; font-weight: 500; color: #334155; }
.col-description { width: 220px; color: #475569; font-size: 13px; }
.col-score { width: 90px; text-align: center; }
.col-rating { width: 90px; text-align: center; }
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

.failure-forward-verification { border-left-color: #8b5cf6; background: #f5f3ff; }
.failure-forward-verification .failure-label { color: #6d28d9; }
.fv-row { margin: 8px 0; display: flex; align-items: baseline; gap: 8px; }
.fv-label { font-weight: 600; color: #6b7280; min-width: 65px; font-size: 13px; }
.fv-expected { background: #fef2f2; color: #dc2626; padding: 1px 8px; border-radius: 4px; font-size: 13px; }
.fv-row code { background: #f1f5f9; color: #334155; padding: 1px 8px; border-radius: 4px; font-size: 13px; }
.fv-suggestion { margin-top: 10px; padding: 8px 12px; background: #fffbeb; border-radius: 6px; color: #92400e; font-size: 12px; }

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
.log-file-list .log-block-label      { color: #6d28d9; }
.log-file-list      { border-left-color: #8b5cf6; background: #f5f3ff; }

/* === Environment toggle === */
#environment-header h2 { cursor: pointer; }
#environment-header.collapsed h2::after { content: " \\25B6"; font-size: 12px; }
#environment-header:not(.collapsed) h2::after { content: " \\25BC"; font-size: 12px; }

/* === Responsive === */
@media (max-width: 768px) {
  body { padding: 10px; font-size: 13px; }
  #results-table { font-size: 12px; }
  #results-table th, #results-table td { padding: 6px 8px; }
  .col-type { width: 60px; }
  .col-skill { width: 100px; }
}
"""


# ═══════════════════════════════════════════════════════════════
#  Footer CSS and builder for HTML report docs hint
# ═══════════════════════════════════════════════════════════════

FOOTER_CSS = """
/* === Docs hint footer === */
.docs-hint {
  margin: 24px 0 12px;
  padding: 16px 20px;
  border-radius: 8px;
  border-left: 4px solid #3b82f6;
  background: #eff6ff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.docs-hint-title {
  font-size: 14px;
  font-weight: 700;
  color: #1e40af;
  margin-bottom: 8px;
}
.docs-hint-body {
  font-size: 13px;
  line-height: 1.7;
  color: #1e293b;
}
.docs-hint-body a {
  color: #2563eb;
  text-decoration: underline;
}
.docs-hint-body a:hover { color: #1d4ed8; }
"""


def _build_footer_html() -> str:
    """生成文档指引提示 footer HTML，统一添加在所有报告底部"""
    return f"""
<style>{FOOTER_CSS}</style>
<div class="docs-hint">
  <div class="docs-hint-title">📖 ST 测试用例开发指引</div>
  <div class="docs-hint-body">
    请查阅 <strong>ST 用例设计与开发规范</strong> 文档了解用例编写与调测方法：<br>
    · 文档路径：<code>tests/system/docs/ST_DESIGN_AND_DEVELOPMENT_GUIDE.md</code><br>
    · GitCode 链接：<a href="https://gitcode.com/cann/cannbot-skills/blob/master/tests/system/docs/ST_DESIGN_AND_DEVELOPMENT_GUIDE.md" target="_blank">https://gitcode.com/cann/cannbot-skills/blob/master/tests/system/docs/ST_DESIGN_AND_DEVELOPMENT_GUIDE.md</a>
  </div>
</div>
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
    # TestEvalsMdStructure
    "test_evals_md_exists": "evals.md 文件存在性",
    "test_evals_md_valid": "evals.md 格式合法性",
    "test_evals_md_has_skill_name": "evals.md 包含 skill_name 字段",
    "test_evals_md_has_evals_list": "evals.md 包含 evals 列表",
    # TestEvalCaseStructure
    "test_eval_cases_have_id": "评测用例具有 id 字段",
    "test_eval_cases_have_name": "评测用例具有 case_name 字段",
    "test_eval_cases_have_prompt": "评测用例具有 prompt 字段",
    "test_eval_cases_have_expected_output": "评测用例具有 expected_output 字段",
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
    # Phase 1: test_skill_basic.py eval_mode 校验
    "test_skill_eval_mode_valid": "eval_mode 字段合法性",
    # Team Phase 1: test_team_basic.py
    "test_team_evals_md_exists": "Team evals.md 文件存在性",
    "test_team_evals_md_valid": "Team evals.md 格式合法性",
    "test_team_evals_md_has_team_name": "evals.md 包含 team_name 字段",
    "test_team_evals_md_has_evals_list": "evals.md 包含 evals 列表",
    "test_team_eval_cases_have_id": "Team 评测用例具有 id 字段",
    "test_team_eval_cases_have_name": "Team 评测用例具有 case_name 字段",
    "test_team_eval_cases_have_prompt": "Team 评测用例具有 prompt 字段",
    "test_team_eval_cases_have_expected_output": "Team 评测用例具有 expected_output 字段",
    "test_team_eval_cases_expectations_format": "Team expectations 字段格式合法",
    "test_team_eval_ids_are_unique": "Team 用例 ID 唯一性",
    "test_team_eval_ids_are_sequential": "Team 用例 ID 连续递增",
    "test_team_prompt_is_descriptive": "Team prompt 非空（描述性检查）",
    "test_team_expected_output_matches_prompt": "Team expected_output 长度检查",
    "test_team_has_agents_md": "AGENTS.md 文件存在性",
    "test_team_agents_md_has_frontmatter": "AGENTS.md YAML frontmatter 格式",
    "test_team_agents_md_has_required_fields": "AGENTS.md frontmatter 必填字段",
    "test_team_has_plugin_json": "plugin.json 文件存在性",
    "test_team_plugin_json_valid": "plugin.json 格式合法性",
    "test_team_has_init_sh": "init.sh 文件存在性",
    "test_team_eval_mode_valid": "Team eval_mode 字段合法性",
    # Team Phase 2: test_team_evals.py
    "test_team_eval_case": "Team AI 语义评测",
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
    # 在 "AssertionError" 之后搜索 "Reviewer reason"（断言报错消息，非源代码）
    after_assert = longrepr.split("AssertionError", 1)[-1] if "AssertionError" in longrepr else longrepr
    reason_match = re.search(
        r'Reviewer reason:\s*(.+?)(?:\n--- AI Response|\n--- End AI Response|\Z)',
        after_assert, re.DOTALL
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
    # 新版中文格式: [contains] 期望输出中包含 "xxx"，但未找到
    if "[contains]" in longrepr or "[not_contains]" in longrepr:
        pm = re.search(r'\[(?:not_)?contains\]\s*(.+?)(?:\n|---|\Z)', longrepr, re.DOTALL)
        if pm:
            msg = html_mod.escape(pm.group(1).strip())
            return (
                f'<div class="failure-block failure-pattern">\n'
                f'  <div class="failure-label">✖ 模式匹配失败 — Eval {eval_id}</div>\n'
                f'  <div class="failure-content">{msg}</div>\n</div>'
            )
    return None


def _extract_expected_skill_name(clean: str) -> str:
    m = re.search(r'期望激活 skill "([^"]+)"', clean)
    return m.group(1) if m else "?"


def _extract_actual_skills(clean: str) -> List[str]:
    """从清理后的失败消息中提取实际加载 skill 列表（去重保持顺序）"""
    block = re.search(
        r'(?:实际加载了以下 skill|loaded the following skills)[：:]\s*\n'
        r'((?:\s*[-*]\s*\S+\s*\n?)+)',
        clean,
    )
    if not block:
        return []
    actual: List[str] = []
    seen: Set[str] = set()
    for line in block.group(1).strip().split('\n'):
        skill = re.sub(r'^\s*[-*]\s*', '', line.strip()).strip()
        if skill and skill not in seen:
            actual.append(skill)
            seen.add(skill)
    return actual


def _extract_suggestion(clean: str) -> str:
    m = re.search(r'(请检查[^。\n]*[。]?)', clean)
    return m.group(0).rstrip('"\'') if m else ""


def _render_actual_skills_row(actual_skills: List[str]) -> str:
    if not actual_skills:
        return ('<div class="fv-row"><span class="fv-label">实际加载</span>'
                '<em>未加载任何 skill</em></div>')
    skills_html = ', '.join(
        f'<code>{html_mod.escape(s)}</code>' for s in actual_skills
    )
    return (f'<div class="fv-row"><span class="fv-label">实际加载</span>'
            f'{skills_html}</div>')


def _parse_skill_activated_block(longrepr: str, eval_id: str) -> Optional[str]:
    """解析正向看护 [skill_activated] 失败块"""
    if "正向看护失败" not in longrepr and "[skill_activated]" not in longrepr:
        return None

    clean = re.sub(r'^E\s{3,}', '', longrepr, flags=re.MULTILINE)
    expected = _extract_expected_skill_name(clean)
    actual_skills = _extract_actual_skills(clean)
    suggestion = _extract_suggestion(clean)

    parts = [
        f'<div class="failure-block failure-forward-verification">\n'
        f'  <div class="failure-label">✖ 正向看护失败 — Eval {eval_id}</div>\n'
        f'  <div class="failure-content">',
        f'<div class="fv-row"><span class="fv-label">期望激活</span>'
        f'<code class="fv-expected">{html_mod.escape(expected)}</code></div>',
        _render_actual_skills_row(actual_skills),
    ]
    if suggestion:
        parts.append(f'<div class="fv-suggestion">{html_mod.escape(suggestion)}</div>')
    parts.append('</div>\n</div>')
    return '\n'.join(parts)


def _parse_execution_error_block(longrepr: str, eval_id: str) -> Optional[str]:
    """解析执行错误块"""
    if "opencode run failed" not in longrepr and "review session error" not in longrepr:
        return None
    # 仅在真正的断言错误行中匹配，排除 traceback 源码上下文中的误匹配
    msg_match = re.search(
        r'(?:AssertionError|E\s{3,})(?:\s*:\s*)?'
        r'(?:opencode run failed|review session error)\s*[-:]\s*(.+?)'
        r'(?:\n---|\nassert|\nE\s|\Z)',
        longrepr, re.DOTALL
    )
    if not msg_match:
        return None
    msg = html_mod.escape(msg_match.group(1).strip())
    return (
        f'<div class="failure-block failure-error">\n'
        f'  <div class="failure-label">✖ 执行错误 — Eval {eval_id}</div>\n'
        f'  <div class="failure-content">{msg}</div>\n'
        f'</div>'
    )


def _parse_token_exceeded_block(longrepr: str, eval_id: str) -> Optional[str]:
    """解析 token 超限错误块"""
    if "token" not in longrepr.lower() or "超过上限" not in longrepr:
        return None
    actual_match = re.search(r'token\s*消耗\s*\((\d+)\)', longrepr)
    max_match = re.search(r'超过上限\s*\((\d+)\)', longrepr)
    actual = actual_match.group(1) if actual_match else "?"
    limit = max_match.group(1) if max_match else "?"
    return (
        f'<div class="failure-block failure-error">\n'
        f'  <div class="failure-label">Token 超限 — Eval {eval_id}</div>\n'
        f'  <div class="failure-content">'
        f'实际消耗 <strong>{actual}</strong> tokens，'
        f'超过上限 <strong>{limit}</strong> tokens</div>\n'
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
    skip_patterns = (
        'assert prompt,', 'assert success,',
        'raise AssertionError(', 'Failed: ',
    )
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(p) for p in skip_patterns):
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


def _parse_failure_to_html(longrepr: str, eval_id: str = "?") -> str:
    """解析断言失败文本，提取结构化信息生成 HTML"""
    if not longrepr:
        return ""

    blocks = []

    # 从消息中提取 eval_id（如 "(Eval 7)" 内嵌格式）
    if eval_id == "?":
        m = re.search(r'\(Eval (\d+)\)', longrepr)
        if m:
            eval_id = m.group(1)
        else:
            m = re.search(r'Eval (\d+)[,:]', longrepr)
            if m:
                eval_id = m.group(1)

    for parser in (
        _parse_skill_activated_block,
        _parse_reviewer_reason_block,
        _parse_pattern_block,
        _parse_token_exceeded_block,
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


def _format_dimension_label(dim_scores: Dict[str, int]) -> str:
    """构建维度分数展示标签字符串。

    格式: " | 覆盖度: 38/40 ✓ | 准确性: 27/30 ✓ | 质量: 15/20 ✓ | Token: 8/10 ✓"
    """
    if not dim_scores:
        return ""
    dim_parts = []
    for dim in DIMENSION_ORDER:
        s = dim_scores.get(dim)
        max_ = DIMENSION_MAX_SCORES.get(dim, "?")
        thresh = DEFAULT_DIMENSION_THRESHOLDS.get(dim, 0)
        if s is not None:
            mark = "✓" if s >= thresh else "✗"
            dim_parts.append(f"{dim}: {s}/{max_} {mark}")
    return " | " + " | ".join(dim_parts) if dim_parts else ""


def _build_review_html_from_template(template_path: Path) -> tuple:
    """读取并解析 review-template.md，构建评审结果 HTML 卡片。

    Returns:
        (review_html: str, score: int | None, dim_scores: dict) 元组。
    """
    if not template_path.exists():
        return "", None, {}
    try:
        review_content = template_path.read_text(encoding="utf-8")
        result = parse_review_md(review_content)
    except (IOError, OSError):
        logger.warning("Failed to read review template from %s", template_path)
        return "", None, {}

    status = result.get("status", "fail")
    score = result.get("score")
    reason = result.get("reason", "")
    dim_scores = parse_dimension_scores(result.get("dimensions"))

    cls = "log-review-pass" if status == "pass" else "log-review-fail"
    label = "评测通过" if status == "pass" else "评测未通过"
    if score is not None:
        label += f" ({score}/100)"
    label += _format_dimension_label(dim_scores)

    review_html = _build_log_block(label, reason if reason else status, cls)
    return review_html, score, dim_scores


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


def _build_phase2_html_from_md(skill_name: str, eval_id):
    """从 sandbox 目录下的 review-template.md 解析评测结果，生成 HTML 卡片。

    直接读取评审 Agent 填写后的 MD 模板文件解析评审结果。

    Args:
        skill_name: skill 名称。
        eval_id: 评测用例 ID。

    Returns:
        (html: str, score: int | None, dim_scores: dict) 元组。
    """
    sandbox_dir = SANDBOX_DIR / f"{skill_name}_eval_{eval_id}"

    # ── 读取填写后的 review-template.md 并构建评审卡片 ──
    template_path = sandbox_dir / "review-template.md"
    review_html, score, dim_scores = _build_review_html_from_template(template_path)

    # ── 读取执行 session JSON（用于 Prompt、思考过程、回复展示）──
    ses_file = sandbox_dir / "logs" / f"{skill_name}_case_{eval_id}_ses.json"
    if not ses_file.exists():
        ses_file = LOGS_DIR / f"{skill_name}_case_{eval_id}_ses.json"

    ses_data = _load_json_file(ses_file)
    ses_messages = ses_data.get("messages", [])
    if not ses_messages and "raw_output" in ses_data:
        try:
            raw = json.loads(ses_data["raw_output"])
            ses_messages = raw.get("messages", [])
        except (json.JSONDecodeError, TypeError):
            pass
    session_blocks = _extract_session_blocks(ses_messages)

    # ── 从执行 session 提取预期要点用于展示 ──
    expected_block = None
    if ses_messages:
        user_parts = ses_messages[0].get("parts", [])
        prompt_text = _get_text_from_parts(user_parts)
        m = re.search(
            r'###\s+预期回复应覆盖的要点\s*\n(.*?)(?:\n###|\Z)',
            prompt_text, re.DOTALL
        )
        if m and m.group(1).strip():
            expected_block = _build_log_block("预期要点", m.group(1), "log-expected-output")

    # ── 组装 HTML 卡片 ──
    blocks = session_blocks[:1]
    if expected_block:
        blocks.append(expected_block)
    if review_html:
        blocks.append(review_html)
    if len(session_blocks) > 1:
        blocks.extend(session_blocks[1:])

    return '\n'.join(blocks), score, dim_scores


def _rating_for_score(score):
    """根据评测得分返回中文质量评级。

    Returns:
        (label: str, css_class: str)。label 为空字符串表示无评级。
    """
    if score is None:
        return ("", "rating-na")
    if score >= 90:
        return ("卓越", "rating-s")
    elif score >= 80:
        return ("优秀", "rating-a")
    elif score >= 70:
        return ("良好", "rating-b")
    elif score >= 60:
        return ("警告", "rating-c")
    else:
        return ("错误", "rating-d")


# ═══════════════════════════════════════════════════════════════
#  pytest-html hooks
# ═══════════════════════════════════════════════════════════════

def _format_eval_score_cell(report):
    """从 report 中提取评测分数，返回 (score, score_html)。"""
    score = getattr(report, '_eval_score', None)
    dim_scores = getattr(report, '_eval_dim_scores', None)
    if score is None:
        # Fallback for xdist: _eval_score not serialized between workers
        for key, val in getattr(report, 'user_properties', []) or []:
            if key == 'eval_score':
                score = val
            elif key == 'eval_dim_scores':
                dim_scores = json.loads(val)
    if score is not None:
        if score >= 80:
            score_cls = "score-high"
        elif score >= 60:
            score_cls = "score-mid"
        else:
            score_cls = "score-low"
        tooltip = _format_dimension_label(dim_scores).lstrip(" | ")
        score_html = (
            f'<span class="score-badge {score_cls}"'
            f' title="{html_mod.escape(tooltip)}"'
            f'>{score}</span>'
        )
    else:
        score_html = f'<span class="score-badge score-na">&mdash;</span>'
    return score, score_html


def _format_rating_cell(score):
    """根据分数生成质量评级 HTML。"""
    rating_label, rating_cls = _rating_for_score(score)
    if rating_label:
        return f'<span class="rating-badge {rating_cls}">{rating_label}</span>'
    return f'<span class="rating-badge rating-na">&mdash;</span>'


def pytest_html_report_title(report):
    report.title = "Skills & Teams Test Report"


def pytest_html_results_summary(prefix, summary, postfix, session):
    prefix.append(f"<style>{REPORT_CSS}</style>")


def pytest_html_results_table_header(cells):
    cells.insert(1, '<th>类型</th>')
    cells.insert(2, '<th class="sortable" data-column-type="skill">名称</th>')
    cells.insert(3, '<th>描述</th>')
    cells.insert(4, '<th>评测得分</th>')
    cells.insert(5, '<th>质量评级</th>')


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

    # 类型列 — 根据 nodeid 判断是 Skill 还是 Team
    if 'test_team_evals' in report.nodeid:
        target_type = 'Team'
    elif 'test_skill_evals' in report.nodeid:
        target_type = 'Skill'
    else:
        target_type = '—'
    cells.insert(1, f'<td class="col-type">{target_type}</td>')

    # 名称列 — 插入在类型之后、Test 之前
    skill_name = _extract_skill_name(report.nodeid)
    cells.insert(2, f'<td class="col-skill">{html_mod.escape(skill_name)}</td>')

    # 描述列 — 插入在名称之后、Test 之前
    desc = _get_test_description(report.nodeid)
    cells.insert(3, f'<td class="col-description">{html_mod.escape(desc)}</td>')

    # 评测得分列 + 质量评级列
    score, score_html = _format_eval_score_cell(report)
    rating_html = _format_rating_cell(score)
    cells.insert(4, f'<td class="col-score">{score_html}</td>')
    cells.insert(5, f'<td class="col-rating">{rating_html}</td>')

    # Test 列简化 — 提取 nodeid 中的参数部分，去掉文件名和函数名前缀
    # 原格式: test_skill_evals.py::test_eval_case[ascendc-env-check::eval_5]
    # 简化后: ascendc-env-check::eval_5
    if len(cells) > 6:
        cells[6] = re.sub(
            r'>[^<]+\[([^\]]+)\]<',
            r'>\1<',
            cells[6]
        )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    """注入结构化的 extra HTML：失败用例解析断言信息，Phase 2 用例从 logs/ JSON 文件解析"""
    if report.when != "call":
        return

    from pytest_html import extras

    extra_items = list(getattr(report, 'extras', []))

    # 从 nodeid 解析 eval_id（用于 failure_html 和 Phase 2 日志）
    # nodeid 格式: test_skill_evals.py::test_eval_case[skill::eval_N]
    bracket_match = re.search(r'\[.*?::eval_(\d+)\]', report.nodeid)
    eval_id = bracket_match.group(1) if bracket_match else "?"

    if report.failed:
        failure_html = _parse_failure_to_html(
            getattr(report, 'longreprtext', '') or '',
            eval_id=eval_id
        )
        if failure_html:
            extra_items.append(extras.html(failure_html))

    # 从 nodeid 解析 skill_name，读取 logs/ 目录下的 JSON 文件
    skill_name = _extract_skill_name(report.nodeid)

    if skill_name and eval_id:
        # 去重：xdist 并行时 worker 和 master 各触发一次此钩子。
        # master 侧 extras 已反序列化为 dict，需同时兼容 dict 和 Extra 对象。
        has_phase2 = any(
            'log-block' in (item.get('content', '') if isinstance(item, dict) else getattr(item, 'content', ''))
            for item in extra_items
        )
        phase2_html, score, dim_scores = _build_phase2_html_from_md(skill_name, eval_id)
        if phase2_html and not has_phase2:
            extra_items.append(extras.html(phase2_html))

        if score is not None and getattr(report, '_eval_score', None) is None:
            setattr(report, '_eval_score', score)
            setattr(report, '_eval_dim_scores', dim_scores)
            if not hasattr(report, 'user_properties'):
                report.user_properties = []
            report.user_properties.append(("eval_score", score))
            if dim_scores:
                report.user_properties.append(("eval_dim_scores", json.dumps(dim_scores, ensure_ascii=False)))

    if extra_items:
        report.extras = extra_items


# ═══════════════════════════════════════════════════════════════
#  Sandbox isolation fixtures
# ═══════════════════════════════════════════════════════════════

def _platform_matches(ascend_platforms, eval_item):
    """检查 eval 用例是否匹配指定的 Ascend 平台。

    与 evals_parser 中 case 侧的 .upper() 一致做大小写归一化。
    返回 True 表示匹配（应保留用例），False 表示不匹配（应跳过）。
    """
    if not ascend_platforms:
        return True  # 未指定平台过滤 → 保留所有用例
    requested = [p.upper() for p in ascend_platforms]
    case_platforms = eval_item.get("ascend_platforms", [])
    if not case_platforms:
        return False  # 未配置平台 → 跳过（不匹配任何平台）
    return any(p in requested for p in case_platforms)


@pytest.fixture(scope="function")
def sandbox_manager() -> 'SandboxManager':
    """提供沙箱管理器（function 级别，支持并行执行）

    默认使用软链接模式；设置 SKILL_SANDBOX_COPY=1 可切回复制模式。
    """
    from sandbox_manager import SandboxManager
    use_symlink = os.environ.get("SKILL_SANDBOX_COPY", "0") != "1"
    manager = SandboxManager(FRAMEWORK_DIR, use_symlink=use_symlink)
    manager.ensure_sandbox_root()
    yield manager
    # 不在这里清理，由 main.py 统一清理
