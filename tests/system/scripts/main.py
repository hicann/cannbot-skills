# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import logging
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from string import Template
from typing import Callable, List, Dict, Any, Optional, Set, Tuple

from dataclasses import dataclass

from common import (
    load_common_config,
    find_entity_path,
    load_entity_evals,
)
from subprocess_streamer import run_subprocess_streaming

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


@dataclass
class _EntityParams:
    """封装 _identify_changed_entities 的关联参数，减少函数参数个数。"""
    whitelist_key: str
    dirs_key: str
    label: str
    entity_getter: Callable
    evals_check_fn: Callable
    dir_check_fn: Callable


def _find_matching_timeout(evals_data: Dict[str, Any], eval_id: str) -> Optional[int]:
    """在 evals 数据中搜索指定 eval_id 的 timeout。"""
    for e in evals_data.get("evals", []):
        if str(e.get("id")) == str(eval_id):
            t = e.get("timeout")
            if t and isinstance(t, (int, float)):
                return int(t)
    return None


def _search_eval_timeout(target_names: List[str], eval_id: str) -> int:
    """在指定 target 中搜索 eval_id 对应的 timeout。"""
    for name in target_names:
        evals_data = load_entity_evals(name)
        if not evals_data:
            continue
        timeout = _find_matching_timeout(evals_data, eval_id)
        if timeout is not None:
            return timeout
    return 1200


class GateChecker:
    def __init__(self, repo_root: str, changed_files: List[str], eval_id: Optional[str] = None,
                 parallel: str = "1", report_only: bool = False,
                 ascend_platforms: Optional[List[str]] = None,
                 all_mode: bool = False):
        self.repo_root = Path(repo_root).resolve()
        self.changed_files = changed_files
        self.eval_id = eval_id
        self.parallel = parallel
        self.report_only = report_only
        self.ascend_platforms = ascend_platforms or []
        self.all_mode = all_mode
        self.test_skill_dir = self.repo_root / "tests" / "system"
        self.results_dir = self.test_skill_dir / "results"
        self.config = load_common_config()
        self._skip_report_template = self._load_skip_report_template()

    # ── @staticmethod 组 ──────────────────────────────────────────────
    # 公开静态方法在前，私有静态方法在后

    @staticmethod
    def get_skill_dir(skill_name: str) -> Optional[Path]:
        """查找 skill 目录路径。"""
        return find_entity_path(skill_name, "skill_dirs", ("SKILL.md",))

    @staticmethod
    def get_team_dir(team_name: str) -> Optional[Path]:
        """查找 team 目录路径。"""
        return find_entity_path(team_name, "team_dirs",
                                ("AGENTS.md", ".claude-plugin/plugin.json"))

    @staticmethod
    def load_evals(skill_name: str, entity_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """加载评测用例，entity_type 为 None 时自动搜索所有目录。"""
        return load_entity_evals(skill_name, entity_type=entity_type)

    @staticmethod
    def _discover_all() -> Tuple[List[str], List[str]]:
        """扫描 Skill/Team 目录，返回 (skills, teams) 全量列表（受 whitelist 过滤）。"""
        from common import discover_entities_with_evals
        skills = set(discover_entities_with_evals("skill_whitelist", target_type="skill"))
        teams = set(discover_entities_with_evals("team_whitelist", target_type="team"))
        return sorted(skills), sorted(teams)

    @staticmethod
    def _check_evals_file_change(parts: tuple, changed_set: set, entity_getter) -> None:
        """检测 evals 文件变更（新架构：{dir}/{entity}/evals/evals.json）。"""
        if len(parts) < 4:
            return

        # 新架构：{dir}/{skill-name}/evals/evals.json
        if parts[-2] == "evals" and parts[-1] == "evals.json":
            entity_name = parts[-3]  # 提取 skill-name
            if entity_getter(entity_name):
                changed_set.add(entity_name)

    @staticmethod
    def _check_opencode_available() -> bool:
        """预检 opencode CLI 是否可用，避免全部 Phase 2 测试因环境问题失败。"""
        import shutil
        opencode_path = shutil.which("opencode")
        if opencode_path:
            logger.info("opencode 可用: %s", opencode_path)
            return True
        logger.error("opencode CLI 未找到，请确保已安装并添加到 PATH")
        return False

    @staticmethod
    def _resolve_max_eval_timeout(target_names: List[str]) -> int:
        """扫描所有目标的 evals 文件，取所有用例的最大 timeout。
        当未指定 eval_id 时使用，确保外层 pytest 子进程超时不小于最慢的单个用例。
        若无任何 timeout 配置，返回 1200。
        """
        max_timeout = 0
        for name in target_names:
            evals_data = load_entity_evals(name)
            if not evals_data:
                continue
            for e in evals_data.get("evals", []):
                t = e.get("timeout")
                if t and isinstance(t, (int, float)):
                    max_timeout = max(max_timeout, int(t))
        return max(max_timeout, 1200)

    @staticmethod
    def _log_result_summary(
        changed_skill_count: int,
        changed_team_count: int,
        candidate_count: int,
        all_passed: bool,
        t_total: float,
    ) -> None:
        """输出评测结果汇总日志。"""
        logger.info("=" * 60)
        if all_passed:
            logger.info("全部通过 — %d skill + %d team, %d 验证完成 (%.1fs)",
                        changed_skill_count, changed_team_count, candidate_count, time.time() - t_total)
        else:
            logger.info("评测存在失败 — %d skill + %d team, %d 验证项 (%.1fs)",
                        changed_skill_count, changed_team_count, candidate_count, time.time() - t_total)
        logger.info("=" * 60)

    # ── 公有实例方法 ──────────────────────────────────────────────────

    def identify_changed_skills(self) -> List[str]:
        return self._identify_changed_entities(_EntityParams(
            "skill_whitelist", "skill_dirs", "skill",
            self.get_skill_dir,
            self._check_evals_file_change,
            self._check_entity_dir_change,
        ))

    def identify_changed_teams(self) -> List[str]:
        return self._identify_changed_entities(_EntityParams(
            "team_whitelist", "team_dirs", "team",
            self.get_team_dir,
            self._check_evals_file_change,
            self._check_entity_dir_change,
        ))

    def run_basic_validation(self, skill_name: str) -> bool:
        """
        skill基本拦截，用例检查
        """
        logger.info("=" * 60)
        logger.info("基础验证 (evals.json 结构检查)")
        logger.info("=" * 60)
        return self._run_basic_validation("test_skill_basic.py", skill_name, "Skill")

    def run_checks(self) -> bool:
        t_total = time.time()
        logger.info("Repository root: %s", self.repo_root)
        if self.all_mode:
            logger.info("模式: --all（自动发现所有 skill/team evals）")
        else:
            logger.info("Changed files: %d", len(self.changed_files))
        if self.report_only:
            logger.info("模式: --report-only (仅重新生成报告，不执行测试)")

        if not self.report_only:
            self._cleanup_previous_run()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        changed_skills: List[str]
        changed_teams: List[str]
        if self.all_mode:
            changed_skills, changed_teams = self._discover_all()
        else:
            changed_skills = self.identify_changed_skills()
            changed_teams = self.identify_changed_teams()

        if not changed_skills and not changed_teams:
            logger.info("没有受影响的 skill 或 team，生成跳过报告。")
            self._generate_skip_report()
            return True

        logger.info("受影响的 skill (%d): %s", len(changed_skills), ', '.join(changed_skills))
        logger.info("受影响的 team (%d): %s", len(changed_teams), ', '.join(changed_teams))

        if not self._check_opencode_available():
            logger.error("opencode 不可用，无法执行 Phase 2 AI 语义评测。")
            logger.error("请确保 opencode 已安装并添加到系统 PATH。")
            return False

        # Phase 1: 逐 target 基础验证
        skill_candidates = self._collect_entities_for_eval(
            changed_skills, "Skill", self.run_basic_validation,
            entity_type="skill",
        )
        team_candidates = self._collect_entities_for_eval(
            changed_teams, "Team", self.run_team_basic_validation,
            entity_type="team",
        )

        if not skill_candidates and not team_candidates:
            logger.info("所有受影响的实体均无评测用例，生成跳过报告。")
            self._generate_skip_report()
            return True

        # Phase 2: 统一的 AI 语义评测，生成一份汇总报告
        all_passed = self._run_unified_eval_pytest(skill_candidates, team_candidates, t_total)

        self._log_result_summary(
            len(changed_skills), len(changed_teams),
            len(skill_candidates) + len(team_candidates),
            all_passed, t_total,
        )
        return all_passed

    def run_team_basic_validation(self, team_name: str) -> bool:
        """Team 静态结构验证（Phase 1）"""
        logger.info("-" * 40)
        logger.info("Team 基础验证: %s", team_name)
        return self._run_basic_validation("test_team_basic.py", team_name, "Team")

    # ── 保护/私有实例方法 ────────────────────────────────────────────

    def _build_eval_pytest_cmd(
        self, skill_names: List[str], team_names: List[str]
    ) -> Tuple[List[str], Path]:
        """构建 pytest 命令和报告路径"""
        cmd = [sys.executable, "-m", "pytest"]
        if skill_names:
            cmd.append("test_skill_evals.py")
        if team_names:
            cmd.append("test_team_evals.py")
        for name in skill_names:
            cmd.extend(["--skill", name])
        for name in team_names:
            cmd.extend(["--team", name])
        if self.eval_id:
            cmd.extend(["--eval-id", self.eval_id])
        if self.ascend_platforms:
            for p in self.ascend_platforms:
                cmd.extend(["--ascend-platform", p])

        report_path = self._build_report_path()
        cmd.extend([f"--html={report_path}", "--self-contained-html", "--tb=short"])
        return cmd, report_path

    def _build_report_path(self) -> Path:
        """构建统一 ST 验证报告路径。"""
        beijing_tz = timezone(timedelta(hours=8))
        timestamp = datetime.now(tz=beijing_tz).strftime("%Y%m%d_%H%M%S")
        platform_prefix = "_".join(self.ascend_platforms) + "_" if self.ascend_platforms else ""
        return self.results_dir / f"{platform_prefix}ST_validation_report_{timestamp}.html"

    def _check_entity_dir_change(self, parts: tuple, changed_set: set, config_key: str) -> None:
        """检测实体目录下的文件变更"""
        for dir_rel in self.config.get(config_key, []):
            dir_parts = Path(dir_rel).parts
            if len(parts) <= len(dir_parts):
                continue
            if parts[:len(dir_parts)] != dir_parts:
                continue
            entity_name = parts[len(dir_parts)]
            entity_dir = self.repo_root / dir_rel / entity_name
            if entity_dir.exists() and entity_dir.is_dir():
                changed_set.add(entity_name)

    def _cleanup_previous_run(self):
        """清除上次运行的 logs 目录，清理 sandboxes 目录内容"""
        import shutil

        # sandboxes：先清理沙箱，避免 logs/results 清空后 sandbox 清理失败导致不一致状态
        sandboxes_dir = self.test_skill_dir / "sandboxes"
        if sandboxes_dir.exists():
            for sandbox in sandboxes_dir.iterdir():
                if not sandbox.is_dir():
                    continue
                try:
                    shutil.rmtree(sandbox)
                    logger.info("[清理] 沙箱: %s", sandbox.name)
                except OSError as e:
                    logger.warning("[清理] 跳过沙箱 %s，删除失败: %s", sandbox.name, e)
            logger.info("[清理] sandboxes/ 内容已清理，目录保留")
        else:
            sandboxes_dir.mkdir(parents=True, exist_ok=True)

        # logs：清空重建；results 保留历史报告不删除
        for dir_rel in ("logs",):
            target = self.test_skill_dir / dir_rel
            if target.exists():
                shutil.rmtree(target)
                target.mkdir()
                logger.info("[清理] %s/ (%s)", dir_rel, target)

    def _collect_entities_for_eval(
        self, changed_entities: List[str], entity_label: str,
        run_basic_validation_fn,
        entity_type: Optional[str] = None,
    ) -> List[str]:
        """Phase 1: 逐实体基础验证，返回通过验证且有 eval 用例的实体列表"""
        if not changed_entities:
            return []

        logger.info("=" * 60)
        logger.info("基础验证 (%s)", entity_label)
        logger.info("=" * 60)

        entities_with_evals: List[str] = []
        for idx, entity_name in enumerate(changed_entities, 1):
            if self.report_only:
                evals_data = self.load_evals(entity_name, entity_type=entity_type)
                if evals_data and evals_data.get("evals", []):
                    entities_with_evals.append(entity_name)
                continue

            logger.info("[%d/%d] %s — 基础验证", idx, len(changed_entities), entity_name)
            if not run_basic_validation_fn(entity_name):
                logger.info("  %s 基础验证失败，跳过", entity_name)
                continue
            evals_data = self.load_evals(entity_name, entity_type=entity_type)
            eval_cases = evals_data.get("evals", []) if evals_data else []
            if eval_cases:
                entities_with_evals.append(entity_name)
                logger.info("  %s — %d 个评测用例", entity_name, len(eval_cases))

        return entities_with_evals

    def _filter_by_whitelist(self, items: Set[str], whitelist_key: str, label: str) -> Set[str]:
        """白名单过滤：仅保留在 whitelist 中的 item，记录被跳过的 item。"""
        whitelist = self.config.get(whitelist_key, [])
        if not whitelist:
            return items
        skipped = sorted(items - set(whitelist))
        if skipped:
            logger.info("白名单过滤 — 跳过的 %s (不在 %s 中): %s", label, whitelist_key, ', '.join(skipped))
        return items & set(whitelist)

    def _generate_skip_report(self) -> None:
        """生成跳过报告（无 skill/team 变更时）。"""
        report_path = self._build_report_path()
        beijing_tz = timezone(timedelta(hours=8))
        now = datetime.now(tz=beijing_tz)
        changed_files_html = "\n".join(
            f"      <li>{file}</li>" for file in self.changed_files
        )
        if self._skip_report_template is None:
            logger.warning("跳过报告模板未加载，跳过报告生成。")
            return
        html = self._skip_report_template.safe_substitute(
            generated_at=now.strftime("%d-%b-%Y at %H:%M:%S"),
            changed_files_html=changed_files_html,
        )
        report_path.write_text(html, encoding='utf-8')
        logger.info("跳过报告已生成: %s", report_path)

    def _get_parallel_workers(self) -> str:
        """
        解析 parallel 参数，返回实际使用的 worker 数量。
        - "1": 顺序执行
        - "auto": CPU 核数 - 1（至少为 1，最大不超过 32）
        - 其他数字: 直接使用
        """
        if self.parallel == "1":
            return "1"
        if self.parallel == "auto":
            cpu_count = os.cpu_count() or 1
            workers = max(1, min(cpu_count - 1, 32))
            return str(workers)
        return self.parallel

    def _identify_changed_entities(
        self, params: _EntityParams,
    ) -> List[str]:
        if self.all_mode:
            all_skills, all_teams = self._discover_all()
            return all_skills if params.label == "skill" else all_teams

        changed = set()
        for file_path in self.changed_files:
            try:
                abs_path = Path(file_path)
                if not abs_path.is_absolute():
                    abs_path = self.repo_root / file_path
                rel_path = abs_path.relative_to(self.repo_root)
                parts = rel_path.parts
            except ValueError:
                continue
            params.evals_check_fn(parts, changed, params.entity_getter)
            params.dir_check_fn(parts, changed, params.dirs_key)
        return sorted(self._filter_by_whitelist(changed, params.whitelist_key, params.label))

    def _load_skip_report_template(self) -> Optional[Template]:
        """加载跳过报告 HTML 模板，文件不存在时返回 None。"""
        template_path = self.test_skill_dir / "config" / "skip-report-template.html"
        if template_path.exists():
            return Template(template_path.read_text(encoding='utf-8'))
        logger.warning("跳过报告模板不存在: %s", template_path)
        return None

    def _resolve_actual_workers(self, skill_names: List[str], team_names: List[str]) -> int:
        """根据实际 eval 用例数 cap worker 数，避免创建过多空闲进程"""
        total_cases = 0
        for name in skill_names + team_names:
            evals_data = self.load_evals(name)
            if evals_data:
                total_cases += len(evals_data.get("evals", []))
        raw_workers = int(self._get_parallel_workers())
        if total_cases > 0 and raw_workers > total_cases:
            return total_cases
        return raw_workers

    def _resolve_eval_timeout(self, target_names: List[str], target_flag: str) -> int:
        """从 evals.json 中读取超时配置。

        当指定了 eval_id → 读取该用例的 Timeout 配置。
        当未指定 eval_id   → 扫描所有目标 evals 文件，取各用例最大 Timeout。
        兜底默认值：1200s。
        """
        if not target_names:
            return 1200
        if not self.eval_id:
            return self._resolve_max_eval_timeout(target_names)
        return _search_eval_timeout(target_names, self.eval_id)

    def _run_basic_validation(self, test_script_name: str, target_name: str,
                               label: str) -> bool:
        """通用 Phase 1 静态验证执行器（消除 skill/team 重复代码）。"""
        test_basic_script = self.test_skill_dir / "scripts" / test_script_name
        if not test_basic_script.exists():
            logger.warning("%s not found, skipping %s basic validation",
                           test_script_name, label)
            return True

        t0 = time.time()
        try:
            returncode, captured_stdout, captured_stderr, timed_out = run_subprocess_streaming(
                [
                    sys.executable, "-m", "pytest",
                    str(test_basic_script),
                    "-v",
                    "--tb=short",
                    "-k", target_name,
                ],
                timeout=120,
                cwd=str(self.test_skill_dir / "scripts"),
                label=f"{label} basic",
            )
            if timed_out:
                logger.error("%s 基础验证超时 (120s)", label)
                return False
            if returncode != 0:
                logger.error("%s 基础验证失败 ✗ (%.1fs)", label, time.time() - t0)
                if captured_stderr.strip():
                    logger.error(captured_stderr)
                return False
            logger.info("%s 基础验证通过 ✓ (%.1fs)", label, time.time() - t0)
            return True
        except Exception as e:
            logger.error("%s basic validation error: %s", label, e)
            return False

    def _run_unified_eval_pytest(
        self, skill_names: List[str], team_names: List[str], t_total: float
    ) -> bool:
        """一次 pytest 运行所有 skill + team 的 eval 用例，生成一份统一 HTML 报告"""
        if not skill_names and not team_names:
            logger.info("无评测用例，跳过 AI 语义评测")
            return True

        cmd, _ = self._build_eval_pytest_cmd(skill_names, team_names)

        actual_workers = self._resolve_actual_workers(skill_names, team_names)
        if actual_workers > 1:
            cmd.extend(["-n", str(actual_workers)])

        env = os.environ.copy()
        if self.report_only:
            env["REPORT_ONLY"] = "1"

        all_targets = skill_names + team_names
        eval_timeout = self._resolve_eval_timeout(all_targets, "--skill")
        try:
            returncode, captured_stdout, captured_stderr, timed_out = run_subprocess_streaming(
                cmd,
                timeout=eval_timeout,
                env=env,
                cwd=str(self.test_skill_dir / "scripts"),
                label="unified eval",
            )
            if timed_out:
                logger.error("统一评测执行超时 (%ds)", eval_timeout)
                return False
            if returncode != 0:
                logger.error("统一评测执行失败 (exit code %d)", returncode)
                if captured_stderr.strip():
                    logger.error(captured_stderr[-2000:])
                return False
            logger.info("统一评测全部通过 ✓ (%.1fs)", time.time() - t_total)
            return True
        except Exception as e:
            logger.error("统一评测执行异常: %s", e)
            return False


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="Gate check for skill testing framework")
    parser.add_argument(
        "--repo-root",
        required=True,
        help="Repository root directory path"
    )
    parser.add_argument(
        "--changed-files",
        nargs="+",
        default=[],
        help="List of changed files (relative or absolute paths)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Run all available eval cases (auto-discover from skill/team evals/ directories)"
    )
    parser.add_argument(
        "--eval-id",
        default=None,
        help="Run specific eval case by ID (forwarded to pytest)"
    )
    parser.add_argument(
        "--parallel", "-p",
        type=str,
        default="1",
        help="Number of parallel pytest workers via pytest-xdist "
             "(default: 1 = sequential, 'auto' = min(CPU cores - 1, 32), "
             "or specify a number like '4')"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        default=False,
        help="仅重新生成 HTML 报告（从已有沙箱 JSON 文件读取数据，不执行测试）"
    )
    parser.add_argument(
        "--eval-model",
        default=None,
        help="指定评测模型名称（如 claude-sonnet-4-20250514），"
             "用于按模型匹配 Max Tokens 预算，默认走 Max Tokens 通用值"
    )
    parser.add_argument(
        "--ascend-platform", action="append",
        default=None,
        help="Filter eval cases by Ascend platform (A2/A3/A5). Repeatable."
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.all and not args.changed_files:
        parser.error("--changed-files is required unless --all is specified")

    # 将 --eval-model 写入环境变量，透传给 pytest 子进程
    if args.eval_model:
        os.environ["EVAL_MODEL"] = args.eval_model

    checker = GateChecker(args.repo_root, args.changed_files, args.eval_id,
                          args.parallel, args.report_only,
                          ascend_platforms=args.ascend_platform,
                          all_mode=args.all)
    success = checker.run_checks()

    archive_logs_and_results(args.repo_root)

    sys.exit(0 if success else 1)


def _add_directory_to_zip(zipf, directory, archive_path, base_dir):
    """Add all files in a directory to the zip archive."""
    if not directory.exists():
        return
    for file_path in directory.rglob("*"):
        if not file_path.is_file() or file_path == archive_path:
            continue
        rel_path = file_path.relative_to(base_dir)
        zipf.write(file_path, rel_path)
    logger.info("  Added %s directory: %s", directory.name, directory)


def archive_logs_and_results(repo_root: str):
    """
    将 logs 和 results 目录打包成压缩文件，放在 logs 目录下，供流水线下载

    Args:
        repo_root: 仓库根目录路径
    """
    repo_path = Path(repo_root).resolve()
    skill_test_framework_dir = repo_path / "tests" / "system"

    logs_dir = skill_test_framework_dir / "logs"
    results_dir = skill_test_framework_dir / "results"

    if not logs_dir.exists() and not results_dir.exists():
        logger.info("No logs or results directory found, skipping archive")
        return

    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_name = f"test_results_{timestamp}.zip"
    archive_path = logs_dir / archive_name

    logger.info("=" * 60)
    logger.info("Archiving logs and results...")
    logger.info("=" * 60)

    try:
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            _add_directory_to_zip(zipf, logs_dir, archive_path, skill_test_framework_dir)
            _add_directory_to_zip(zipf, results_dir, archive_path, skill_test_framework_dir)

        archive_size = archive_path.stat().st_size
        size_mb = archive_size / (1024 * 1024)
        logger.info("  Archive created: %s", archive_path)
        logger.info("  Archive size: %.2f MB", size_mb)

    except Exception as e:
        logger.error("  Error creating archive: %s", e)


if __name__ == "__main__":
    main()
