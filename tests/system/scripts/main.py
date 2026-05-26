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
import json
import logging
import os
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    eval_id: int
    passed: bool
    prompt: str
    expected_output: str
    actual_output: str = ""
    error: str = ""
    expectations: List[str] = field(default_factory=list)
    expectations_met: List[str] = field(default_factory=list)
    expectations_failed: List[str] = field(default_factory=list)


@dataclass
class EvalContext:
    """封装评测执行所需的上下文参数"""
    skill_name: str
    eval_id: int
    prompt: str
    expected_output: str
    expectations: List[str]
    skill_dir: Path


@dataclass
class SkillEvalResult:
    eval_id: int
    passed: bool
    error: str = ""
    actual_output: str = ""


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
        self.results: List[EvalResult] = []

    def get_skill_dir(self, skill_name: str) -> Optional[Path]:
        for skill_dir_rel in self.config.get("skill_dirs", ["skills"]):
            candidate = self.repo_root / skill_dir_rel / skill_name
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    def run_basic_validation(self, skill_name: str) -> bool:
        """
        skill基本拦截，用例检查
        """
        logger.info("=" * 60)
        logger.info("基础验证 (evals.json 结构检查)")
        logger.info("=" * 60)

        test_basic_script = self.test_skill_dir / "scripts" / "test_skill_basic.py"
        if not test_basic_script.exists():
            logger.warning("test_skill_basic.py not found, skipping basic validation")
            return True

        t0 = time.time()

        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "pytest",
                    str(test_basic_script),
                    "-v",
                    "--tb=short",
                    "-k", skill_name,
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
                logger.error("基础验证失败 ✗ (%.1fs)", time.time() - t0)
                if proc.stderr:
                    logger.error(proc.stderr)
                return False

            logger.info("基础验证通过 ✓ (%.1fs)", time.time() - t0)
            return True

        except subprocess.TimeoutExpired:
            logger.error("Basic validation timed out")
            return False
        except Exception as e:
            logger.error("Basic validation error: %s", e)
            return False

    def run_skill_eval(self, skill_name: str) -> EvalResult:
        """
        skill 级别的验证，批量验证
        """

        skill_dir = self.get_skill_dir(skill_name)
        if not skill_dir:
            result = EvalResult(eval_id=0, passed=False, prompt="", expected_output="")
            result.error = f"Skill directory not found: {skill_name}"
            return result
        skill_test_script = self.test_skill_dir / "scripts" / "test_skill_evals.py"
        # 复用单用例结果
        result = EvalResult(eval_id=0, passed=False, prompt="", expected_output="")
        if skill_test_script.exists():
            try:
                env = os.environ.copy()
                env["SKILL_DIR"] = str(skill_dir)
                if self.report_only:
                    env["REPORT_ONLY"] = "1"

                cmd = [sys.executable, "-m", "pytest", str(skill_test_script), "--skill", skill_name]
                if self.eval_id:
                    cmd.extend(["--eval-id", self.eval_id])
                cmd.extend([
                    "--html=" + str(self.results_dir / (skill_name + "_evals_validation.html")),
                    "--self-contained-html"
                ])
                if self._get_parallel_workers() != "1":
                    cmd.extend(["-n", self._get_parallel_workers()])
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    env=env,
                    timeout=1200,
                    cwd=str(skill_dir)
                )
                if proc.returncode == 0:
                    result.passed = True
                else:
                    result.error = proc.stderr or f"Exit code: {proc.returncode}"
            except subprocess.TimeoutExpired:
                result.error = "Test execution timed out"
            except Exception as e:
                result.error = str(e)
        else:
            result.passed = True
            result.actual_output = f"No test script found for {skill_name}, skipping evaluation"
        return result

    def run_single_eval(self, skill_name: str, eval_case: Dict[str, Any]) -> EvalResult:
        """
        按用例级别验证
        """
        eval_id = eval_case.get("id", 0)
        prompt = eval_case.get("prompt", "")
        expected_output = eval_case.get("expected_output", "")
        expectations = eval_case.get("expectations", [])
        result = EvalResult(
            eval_id=eval_id, passed=False, prompt=prompt,
            expected_output=expected_output, expectations=expectations,
        )
        skill_dir = self.get_skill_dir(skill_name)
        if not skill_dir:
            result.error = f"Skill directory not found: {skill_name}"
            return result
        skill_test_script = self.test_skill_dir / "scripts" / "test_skill_evals.py"
        if not skill_test_script.exists():
            result.passed = True
            result.actual_output = f"No test script found for {skill_name}, skipping evaluation"
            return result
        ctx = EvalContext(
            skill_name=skill_name,
            eval_id=eval_id,
            prompt=prompt,
            expected_output=expected_output,
            expectations=expectations,
            skill_dir=skill_dir,
        )
        passed, actual_output, error = self._execute_eval_cmd(ctx)
        result.passed = passed
        result.actual_output = actual_output
        result.error = error
        if passed:
            result.expectations_met = expectations
        else:
            result.expectations_failed = expectations
        return result

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

        if not changed_skills:
            logger.info("没有受影响的 skill，跳过测试。")
            return True

        logger.info("受影响的 skill (%d): %s", len(changed_skills), ', '.join(changed_skills))

        eval_passed, eval_total = self._eval_skills(changed_skills, t_total)

        all_passed = eval_total == 0 or eval_passed == eval_total

        logger.info("=" * 60)
        if all_passed:
            logger.info("全部通过 — %d 个 skill 验证完成 (%.1fs)", eval_total, time.time() - t_total)
        else:
            logger.info("评测存在失败 — %d 个 skill, %d 通过 (%.1fs)",
                        eval_total, eval_passed, time.time() - t_total)
        logger.info("=" * 60)

        return all_passed

    def _eval_skills(self, changed_skills: List[str], t_total: float) -> tuple:
        logger.info("=" * 60)
        logger.info("AI 语义评测")
        logger.info("=" * 60)

        eval_passed = 0
        eval_total = 0

        for idx, skill_name in enumerate(changed_skills, 1):
            if not self.report_only:
                logger.info("[%d/%d] %s — 基础验证", idx, len(changed_skills), skill_name)
                if not self.run_basic_validation(skill_name):
                    logger.info("基础验证失败，终止流程 (%.1fs)", time.time() - t_total)
                    return 0, 0

            evals_data = self.load_evals(skill_name)
            eval_cases = evals_data.get("evals", []) if evals_data else []

            if not eval_cases:
                logger.info("[%d/%d] %s — 无评测用例，跳过", idx, len(changed_skills), skill_name)
                continue

            eval_total += 1
            logger.info("[%d/%d] %s — %d 个评测用例", idx, len(changed_skills), skill_name, len(eval_cases))

            skill_test_script = self.test_skill_dir / "scripts" / "test_skill_evals.py"
            if not skill_test_script.exists():
                logger.info("  test_skill_evals.py 不存在，跳过")
                continue

            self.results = []
            t_skill = time.time()
            result = self.run_skill_eval(skill_name)
            self.results.append(result)
            elapsed = time.time() - t_skill

            if result.passed:
                eval_passed += 1
            status = "✓ 通过" if result.passed else "✗ 失败"
            logger.info("  %s (%.1fs)", status, elapsed)

        return eval_passed, eval_total

    def _execute_eval_cmd(self, ctx: EvalContext) -> tuple:
        """Execute eval pytest command, return (passed, actual_output, error)."""
        skill_test_script = self.test_skill_dir / "scripts" / "test_skill_evals.py"
        env = os.environ.copy()
        env["EVAL_PROMPT"] = ctx.prompt
        env["EVAL_EXPECTED"] = ctx.expected_output
        env["EVAL_EXPECTATIONS"] = json.dumps(ctx.expectations)
        env["SKILL_DIR"] = str(ctx.skill_dir)
        cmd = [
            sys.executable, "-m", "pytest", str(skill_test_script),
            "--skill", ctx.skill_name, "--eval-id", str(ctx.eval_id),
            "--html=" + str(self.results_dir / (ctx.skill_name + "_evals_validation.html")),
            "--self-contained-html",
        ]
        if self._get_parallel_workers() != "1":
            cmd.extend(["-n", self._get_parallel_workers()])
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding='utf-8', errors='replace',
                env=env, timeout=300, cwd=str(ctx.skill_dir),
            )
            if proc.returncode == 0:
                return (True, proc.stdout, "")
            return (False, proc.stdout, proc.stderr or f"Exit code: {proc.returncode}")
        except subprocess.TimeoutExpired:
            return (False, "", "Test execution timed out")
        except Exception as exc:
            return (False, "", str(exc))

    def _cleanup_previous_run(self):
        """清除上次运行的 logs、results 目录，清理 sandboxes 目录内容"""
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

        # logs 和 results：清空重建
        for dir_rel in ("logs", "results"):
            target = self.test_skill_dir / dir_rel
            if target.exists():
                shutil.rmtree(target)
                target.mkdir()
                logger.info("[清理] %s/ (%s)", dir_rel, target)

    def _load_config(self) -> Dict[str, Any]:
        config_path = self.test_skill_dir / "config" / "skill-test.config"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {"skill_dirs": ["skills"]}

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

    args = parser.parse_args()

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
