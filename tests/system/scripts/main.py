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
from typing import List, Dict, Any, Optional

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


class GateChecker:
    def __init__(self, repo_root: str, changed_files: List[str], eval_id: Optional[str] = None,
                 parallel: str = "1", report_only: bool = False):
        self.repo_root = Path(repo_root).resolve()
        self.changed_files = changed_files
        self.eval_id = eval_id
        self.parallel = parallel
        self.report_only = report_only
        self.test_skill_dir = self.repo_root / "tests" / "system"
        self.results_dir = self.test_skill_dir / "results"
        self.evals_cases_dir = self.test_skill_dir / "cases"
        self.config = self._load_config()

    def get_skill_dir(self, skill_name: str) -> Optional[Path]:
        for skill_dir_rel in self.config.get("skill_dirs", ["skills"]):
            candidate = self.repo_root / skill_dir_rel / skill_name
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    def get_team_dir(self, team_name: str) -> Optional[Path]:
        for team_dir_rel in self.config.get("team_dirs", []):
            candidate = self.repo_root / team_dir_rel / team_name
            if candidate.exists() and candidate.is_dir():
                plugin_json = candidate / ".claude-plugin" / "plugin.json"
                agents_md = candidate / "AGENTS.md"
                if plugin_json.exists() and agents_md.exists():
                    return candidate
        return None

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
            proc = subprocess.run(
                [
                    sys.executable, "-m", "pytest",
                    str(test_basic_script),
                    "-v",
                    "--tb=short",
                    "-k", target_name,
                ],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=120,
                cwd=str(self.test_skill_dir / "scripts")
            )
            logger.info(proc.stdout)
            if proc.returncode != 0:
                logger.error("%s 基础验证失败 ✗ (%.1fs)", label, time.time() - t0)
                if proc.stderr:
                    logger.error(proc.stderr)
                return False
            logger.info("%s 基础验证通过 ✓ (%.1fs)", label, time.time() - t0)
            return True
        except subprocess.TimeoutExpired:
            logger.error("%s basic validation timed out", label)
            return False
        except Exception as e:
            logger.error("%s basic validation error: %s", label, e)
            return False

    def run_basic_validation(self, skill_name: str) -> bool:
        """
        skill基本拦截，用例检查
        """
        logger.info("=" * 60)
        logger.info("基础验证 (evals.json 结构检查)")
        logger.info("=" * 60)
        return self._run_basic_validation("test_skill_basic.py", skill_name, "Skill")

    def run_team_basic_validation(self, team_name: str) -> bool:
        """Team 静态结构验证（Phase 1）"""
        logger.info("-" * 40)
        logger.info("Team 基础验证: %s", team_name)
        return self._run_basic_validation("test_team_basic.py", team_name, "Team")

    def _run_eval_pytest(self, test_script_name: str, report_prefix: str,
                          target_flag: str, target_names: List[str]) -> bool:
        """通用 Phase 2 eval pytest 执行器（消除 skill/team 重复代码）。"""
        test_script = self.test_skill_dir / "scripts" / test_script_name
        if not test_script.exists():
            logger.info("  %s 不存在，跳过", test_script_name)
            return True

        cmd = [sys.executable, "-m", "pytest", str(test_script)]
        for name in target_names:
            cmd.extend([target_flag, name])
        if self.eval_id:
            cmd.extend(["--eval-id", self.eval_id])

        beijing_tz = timezone(timedelta(hours=8))
        timestamp = datetime.now(tz=beijing_tz).strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f"{report_prefix}_{timestamp}.html"
        cmd.extend([f"--html={report_path}", "--self-contained-html"])

        if self._get_parallel_workers() != "1":
            cmd.extend(["-n", self._get_parallel_workers()])

        env = os.environ.copy()
        if self.report_only:
            env["REPORT_ONLY"] = "1"

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env,
                timeout=1200,
                cwd=str(self.test_skill_dir / "scripts"),
            )
            if proc.returncode == 0:
                return True
            logger.error("%s 评测执行失败 (exit code %d)", report_prefix, proc.returncode)
            if proc.stderr:
                logger.error(proc.stderr[-2000:])
            return False
        except subprocess.TimeoutExpired:
            logger.error("%s 评测执行超时 (1200s)", report_prefix)
            return False

    def run_skills_eval(self, skill_names: List[str]) -> bool:
        """一次 pytest 运行所有 skill 的 eval 用例，生成一份统一 HTML 报告"""
        return self._run_eval_pytest(
            "test_skill_evals.py", "evals_validation", "--skill", skill_names,
        )

    def run_teams_eval(self, team_names: List[str]) -> bool:
        """一次 pytest 运行所有 team 的 eval 用例，生成独立 HTML 报告"""
        return self._run_eval_pytest(
            "test_team_evals.py", "team_evals_validation", "--team", team_names,
        )

    def identify_changed_skills(self) -> List[str]:
        changed_skills = set()

        for file_path in self.changed_files:
            try:
                abs_path = Path(file_path)
                if not abs_path.is_absolute():
                    abs_path = self.repo_root / file_path
                rel_path = abs_path.relative_to(self.repo_root)
                parts = rel_path.parts
            except ValueError:
                continue

            self._check_evals_file_change(parts, changed_skills)
            self._check_skill_dir_change(parts, changed_skills)

        # 白名单过滤：仅在白名单中的 skill 才会触发评测
        skill_whitelist = self.config.get("skill_whitelist", [])
        if skill_whitelist:
            skipped = sorted(changed_skills - set(skill_whitelist))
            if skipped:
                logger.info("白名单过滤 — 跳过的 skill (不在 skill_whitelist 中): %s", ', '.join(skipped))
            changed_skills = changed_skills & set(skill_whitelist)

        return sorted(list(changed_skills))

    def identify_changed_teams(self) -> List[str]:
        changed_teams = set()

        for file_path in self.changed_files:
            try:
                abs_path = Path(file_path)
                if not abs_path.is_absolute():
                    abs_path = self.repo_root / file_path
                rel_path = abs_path.relative_to(self.repo_root)
                parts = rel_path.parts
            except ValueError:
                continue

            self._check_team_evals_file_change(parts, changed_teams)
            self._check_team_dir_change(parts, changed_teams)

        # 白名单过滤
        team_whitelist = self.config.get("team_whitelist", [])
        if team_whitelist:
            skipped = sorted(changed_teams - set(team_whitelist))
            if skipped:
                logger.info("白名单过滤 — 跳过的 team (不在 team_whitelist 中): %s", ', '.join(skipped))
            changed_teams = changed_teams & set(team_whitelist)

        return sorted(list(changed_teams))

    def load_evals(self, skill_name: str) -> Optional[Dict[str, Any]]:
        evals_path = self.evals_cases_dir / f"{skill_name}_evals.md"
        if not evals_path.exists():
            return None
        try:
            from evals_parser import parse_evals_md
            return parse_evals_md(evals_path)
        except Exception as e:
            logger.error("Error loading evals for %s: %s", skill_name, e)
            return None

    def run_checks(self) -> bool:
        t_total = time.time()
        logger.info("Repository root: %s", self.repo_root)
        logger.info("Changed files: %d", len(self.changed_files))
        if self.report_only:
            logger.info("模式: --report-only (仅重新生成报告，不执行测试)")

        if not self.report_only:
            self._cleanup_previous_run()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        changed_skills = self.identify_changed_skills()
        changed_teams = self.identify_changed_teams()

        if not changed_skills and not changed_teams:
            logger.info("没有受影响的 skill 或 team，跳过测试。")
            return True

        logger.info("受影响的 skill (%d): %s", len(changed_skills), ', '.join(changed_skills))
        logger.info("受影响的 team (%d): %s", len(changed_teams), ', '.join(changed_teams))

        skill_passed, skill_total = self._eval_skills(changed_skills, t_total)
        team_passed, team_total = self._eval_teams(changed_teams, t_total)

        total_passed = skill_passed + team_passed
        total_all = skill_total + team_total
        all_passed = total_all == 0 or total_passed == total_all

        logger.info("=" * 60)
        if all_passed:
            logger.info("全部通过 — %d skill + %d team, %d 验证完成 (%.1fs)",
                        skill_total, team_total, total_all, time.time() - t_total)
        else:
            logger.info("评测存在失败 — %d skill + %d team, %d/%d 通过 (%.1fs)",
                        skill_total, team_total, total_passed, total_all, time.time() - t_total)
        logger.info("=" * 60)

        return all_passed

    def _eval_skills(self, changed_skills: List[str], t_total: float) -> tuple:
        logger.info("=" * 60)
        logger.info("AI 语义评测")
        logger.info("=" * 60)

        # Phase 1: 逐 skill 静态验证（无报告生成，保持逐 skill）
        for idx, skill_name in enumerate(changed_skills, 1):
            if not self.report_only:
                logger.info("[%d/%d] %s — 基础验证", idx, len(changed_skills), skill_name)
                if not self.run_basic_validation(skill_name):
                    logger.info("基础验证失败，终止流程 (%.1fs)", time.time() - t_total)
                    return 0, len(changed_skills)

        # Phase 2: 合并 AI 语义评测，生成一份报告
        skills_with_evals = []
        for skill_name in changed_skills:
            evals_data = self.load_evals(skill_name)
            eval_cases = evals_data.get("evals", []) if evals_data else []
            if eval_cases:
                skills_with_evals.append(skill_name)
                logger.info("  %s — %d 个评测用例", skill_name, len(eval_cases))

        if not skills_with_evals:
            logger.info("无评测用例的 skill，跳过 AI 语义评测")
            return 0, 0

        eval_total = len(skills_with_evals)
        logger.info("[合并] %d 个 skill 合并评测", eval_total)

        passed = self.run_skills_eval(skills_with_evals)
        if passed:
            logger.info("[合并] 全部通过 ✓ (%.1fs)", time.time() - t_total)
        else:
            logger.info("[合并] 存在失败 ✗ (%.1fs)", time.time() - t_total)

        return (eval_total if passed else 0), eval_total

    def _eval_teams(self, changed_teams: List[str], t_total: float) -> tuple:
        if not changed_teams:
            return 0, 0

        logger.info("=" * 60)
        logger.info("Team AI 语义评测")
        logger.info("=" * 60)

        # Phase 1: 逐 team 静态验证，失败的不进入 Phase 2
        passed_teams: List[str] = []
        for idx, team_name in enumerate(changed_teams, 1):
            if not self.report_only:
                logger.info("[%d/%d] %s — Team 基础验证", idx, len(changed_teams), team_name)
                if not self.run_team_basic_validation(team_name):
                    logger.info("Team 基础验证失败，跳过该 team 的 AI 评测 (%.1fs)", time.time() - t_total)
                    continue
            passed_teams.append(team_name)

        # Phase 2: 合并 AI 语义评测，仅对通过 Phase 1 的 team
        teams_with_evals = []
        for team_name in passed_teams:
            evals_data = self.load_evals(team_name)
            eval_cases = evals_data.get("evals", []) if evals_data else []
            if eval_cases:
                teams_with_evals.append(team_name)
                logger.info("  %s — %d 个评测用例", team_name, len(eval_cases))

        if not teams_with_evals:
            # 若所有 team 在 Phase 1 均失败，返回失败计数而非 (0,0) 以避免静默吞掉失败
            total_teams = len(changed_teams)
            logger.info("无评测用例的 team，跳过 Team AI 语义评测")
            return 0, total_teams if total_teams > 0 else 0

        eval_total = len(teams_with_evals)
        logger.info("[合并] %d 个 team 合并评测", eval_total)

        passed = self.run_teams_eval(teams_with_evals)
        if passed:
            logger.info("[合并] Team 全部通过 ✓ (%.1fs)", time.time() - t_total)
        else:
            logger.info("[合并] Team 存在失败 ✗ (%.1fs)", time.time() - t_total)

        return (eval_total if passed else 0), eval_total

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

    def _load_config(self) -> Dict[str, Any]:
        config_path = self.test_skill_dir / "config" / "st-test.config"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {"skill_dirs": ["skills"], "skill_whitelist": [],
                "team_dirs": [], "team_whitelist": []}

    def _check_evals_file_change(self, parts: tuple, changed_skills: set) -> None:
        """检测集中式 evals 文件变更"""
        if len(parts) < 3 or parts[:3] != ("tests", "system", "cases"):
            return
        filename = parts[-1]
        if filename.endswith("_evals.md"):
            skill_name = filename[:-len("_evals.md")]
            if self.get_skill_dir(skill_name):
                changed_skills.add(skill_name)

    def _check_skill_dir_change(self, parts: tuple, changed_skills: set) -> None:
        """检测 skill 目录下的文件变更"""
        for skill_dir_rel in self.config.get("skill_dirs", ["skills"]):
            dir_parts = Path(skill_dir_rel).parts
            if len(parts) <= len(dir_parts):
                continue
            if parts[:len(dir_parts)] != dir_parts:
                continue
            skill_name = parts[len(dir_parts)]
            skill_dir = self.repo_root / skill_dir_rel / skill_name
            if skill_dir.exists() and skill_dir.is_dir():
                changed_skills.add(skill_name)

    def _check_team_evals_file_change(self, parts: tuple, changed_teams: set) -> None:
        """检测 team evals 文件变更"""
        if len(parts) < 3 or parts[:3] != ("tests", "system", "cases"):
            return
        filename = parts[-1]
        if filename.endswith("_evals.md"):
            candidate_name = filename[:-len("_evals.md")]
            if self.get_team_dir(candidate_name):
                changed_teams.add(candidate_name)

    def _check_team_dir_change(self, parts: tuple, changed_teams: set) -> None:
        """检测 team 目录下的文件变更"""
        for team_dir_rel in self.config.get("team_dirs", []):
            dir_parts = Path(team_dir_rel).parts
            if len(parts) <= len(dir_parts):
                continue
            if parts[:len(dir_parts)] != dir_parts:
                continue
            team_name = parts[len(dir_parts)]
            team_dir = self.repo_root / team_dir_rel / team_name
            if team_dir.exists() and team_dir.is_dir():
                changed_teams.add(team_name)

    def _get_parallel_workers(self) -> str:
        """
        解析 parallel 参数，返回实际使用的 worker 数量。
        - "1": 顺序执行
        - "auto": CPU 核数 - 1（至少为 1）
        - 其他数字: 直接使用
        """
        if self.parallel == "1":
            return "1"
        if self.parallel == "auto":
            cpu_count = os.cpu_count() or 1
            workers = max(1, cpu_count - 1)
            return str(workers)
        return self.parallel


def main():
    parser = argparse.ArgumentParser(description="Gate check for skill testing framework")
    parser.add_argument(
        "--repo-root",
        required=True,
        help="Repository root directory path"
    )
    parser.add_argument(
        "--changed-files",
        nargs="+",
        required=True,
        help="List of changed files (relative or absolute paths)"
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
             "(default: 1 = sequential, 'auto' = CPU cores - 1, "
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

    args = parser.parse_args()

    # 将 --eval-model 写入环境变量，透传给 pytest 子进程
    if args.eval_model:
        os.environ["EVAL_MODEL"] = args.eval_model

    checker = GateChecker(args.repo_root, args.changed_files, args.eval_id,
                          args.parallel, args.report_only)
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
