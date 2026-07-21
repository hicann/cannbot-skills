#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Workspace Initializer - 从 CAKE2 output 初始化 workspace inputs

设计原则:
1. 使用 PathManager 进行所有路径管理（配置驱动）
2. 模块化设计：检测、拷贝、验证、报告职责分离
3. 支持 dry-run、选择性拷贝、过滤规则
4. 完整的错误处理和详细日志
5. 向后兼容（支持旧数据迁移）

新的 workspace 结构:
inputs/{op}/
├── code/
│   ├── op_host/
│   └── op_kernel/
└── profiling/
    └── op_summary.csv

示例:
    python3 init_workspace.py
    python3 init_workspace.py --op fastgelu
    python3 init_workspace.py --dry-run
    python3 init_workspace.py --skip-profiling
"""

import argparse
import fnmatch
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# 动态导入 path_manager (确保在同一目录)
sys.path.insert(0, str(Path(__file__).parent))
from path_manager import PathManager


# ========== 数据结构 ==========


class CopyMode(Enum):
    """拷贝模式"""
    FULL = "full"  # 拷贝所有（code + profiling）
    CODE_ONLY = "code_only"  # 仅拷贝代码
    PROFILING_ONLY = "profiling_only"  # 仅拷贝 profiling


@dataclass
class CopyStats:
    """文件拷贝统计"""
    total_files: int = 0
    copied_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    total_size_bytes: int = 0

    def merge(self, other: "CopyStats"):
        """合并统计信息"""
        self.total_files += other.total_files
        self.copied_files += other.copied_files
        self.skipped_files += other.skipped_files
        self.failed_files += other.failed_files
        self.total_size_bytes += other.total_size_bytes


@dataclass
class OperatorResult:
    """单个算子的处理结果"""
    op_name: str
    success: bool
    stats: CopyStats = field(default_factory=CopyStats)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source_dir: Optional[Path] = None
    target_dir: Optional[Path] = None
    code_hash: Optional[str] = None


@dataclass
class InitConfig:
    """初始化配置"""
    dry_run: bool = False
    copy_mode: CopyMode = CopyMode.FULL
    exclude_patterns: Set[str] = field(default_factory=lambda: {"CMakeLists*", "*.o", "*.so", "build/"})
    overwrite_existing: bool = False
    verify_copy: bool = True
    compute_hash: bool = False


# ========== 工具函数 ==========


def log_info(msg: str):
    """信息日志"""
    print(f"ℹ️  {msg}", flush=True)


def log_success(msg: str):
    """成功日志"""
    print(f"✅ {msg}", flush=True)


def log_warning(msg: str):
    """警告日志"""
    print(f"⚠️  {msg}", flush=True)


def log_error(msg: str):
    """错误日志"""
    print(f"❌ {msg}", flush=True)


def compute_dir_hash(path: Path) -> str:
    """计算目录内容的 MD5 哈希（用于检测代码变更）"""
    hasher = hashlib.md5()

    for root, _, files in sorted(os.walk(path)):
        for file_name in sorted(files):
            file_path = Path(root) / file_name
            try:
                with open(file_path, "rb") as f:
                    hasher.update(f.read())
            except Exception as e:
                logger.debug(f"Skipping unreadable file during hash: {file_path}: {e}")
                continue

    return hasher.hexdigest()[:8]


# ========== 核心模块 ==========


class SourceDetector:
    """源文件检测器 - 负责在 CAKE2/output 下定位源文件"""

    def __init__(self, pm: PathManager):
        self.pm = pm

    def detect_output_dir(self) -> Path:
        """检测 CAKE2 output 目录"""
        cake2_root = self.pm.get_cake2_root()
        output_dir = cake2_root / "output"

        if not output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")

        return output_dir

    @staticmethod
    def list_operators(output_dir: Path, filter_name: Optional[str] = None) -> List[Path]:
        """列出所有算子目录"""
        op_dirs = [p for p in output_dir.iterdir() if p.is_dir()]

        if filter_name:
            # 支持大小写不敏感匹配
            op_dirs = [
                p for p in op_dirs
                if p.name.lower() == filter_name.lower()
            ]

            if not op_dirs:
                raise FileNotFoundError(f"Operator not found: {filter_name}")

        return sorted(op_dirs)

    @staticmethod
    def find_code_dirs(op_dir: Path) -> Dict[str, Optional[Path]]:
        """
        查找 op_host 和 op_kernel 目录

        Returns:
            {"op_host": Path | None, "op_kernel": Path | None}
        """
        result = {"op_host": None, "op_kernel": None}

        for sub_name in ["op_host", "op_kernel"]:
            # 递归搜索（支持深层嵌套）
            for candidate in op_dir.rglob(sub_name):
                if candidate.is_dir():
                    result[sub_name] = candidate
                    break

        return result

    def find_best_profiling_csv(self, op_dir: Path) -> Optional[Path]:
        """查找最相关的 op_summary CSV 文件（泛化路径搜索 + 评分选最优）

        选择策略（按优先级）：
        1. 优先从 results.json 找 speedup 最差的 case，定向搜索其 custom/ profiling
        2. Fallback：全树搜索所有包含 op_summary 字样的 CSV，按评分规则选最优
           - 优先 custom/（+10）排除 ref/（-10），排除 origin_data/（-100）
           - 时间戳降序，路径越浅越优先

        注意：此函数不是按 mtime 选"最新"文件，而是按语义相关性和目录结构评分选
        最优文件（custom > ref，排除 origin_data，时间戳仅用于同分时的二次排序）。

        典型支持的目录结构：
          output/{op}/profiling/op_summary.csv                        (标准平铺格式)
          output/{op}/_msprof_work/{case}/custom/.../op_summary_*.csv (msprof 深嵌套)
          output/{op}/msprof_report/{case}/op_summary_*.csv           (旧版 msprof 格式)
          output/{op}/{OpCustom}/profiling/op_summary_*.csv           (Custom 子目录格式)
        """
        # === 策略 1：results.json 驱动的语义选择 ===
        worst_case = self._find_worst_case_name(op_dir)
        if worst_case:
            # 在 _msprof_work/{worst_case}/custom/ 下找 op_summary
            case_dir = op_dir / "_msprof_work" / worst_case / "custom"
            if case_dir.exists():
                candidates = []
                for p in case_dir.rglob("*.csv"):
                    if "op_summary" in p.name.lower() and "origin_data" not in str(p):
                        candidates.append(p)
                if candidates:
                    best = max(candidates, key=lambda p: p.stat().st_mtime)
                    log_info(f"  Selected profiling (worst-case targeted): {best}")
                    return best
            log_warning(f"  Worst case '{worst_case}' has no custom profiling, falling back to global search")

        # === 策略 2：全树泛化搜索 ===
        candidates = [
            p for p in op_dir.rglob("*.csv")
            if "op_summary" in p.name.lower()
        ]

        if not candidates:
            return None

        def score(path: Path) -> tuple:
            parts = str(path)
            origin_penalty = -100 if "origin_data" in parts else 0
            if "/custom/" in parts or "\\custom\\" in parts:
                custom_score = 10
            elif "/ref/" in parts or "\\ref\\" in parts:
                custom_score = -10
            else:
                custom_score = 0
            tokens = re.findall(r"(20\d{6}_\d{6}|20\d{12})", parts)
            mtime_ts = int(path.stat().st_mtime)
            if tokens:
                try:
                    ts = int(tokens[-1].replace("_", ""))
                except ValueError:
                    ts = mtime_ts
            else:
                ts = mtime_ts
            depth_penalty = -len(path.parts)
            return (origin_penalty + custom_score, ts, depth_penalty)

        best = max(candidates, key=score)
        log_info(f"  Selected profiling (fallback global search): {best}")
        return best

    @staticmethod
    def _find_worst_case_name(op_dir: Path) -> Optional[str]:
        """从 results.json 找出 speedup 最差的 case 名（最有优化价值）"""
        results_json = op_dir / "results.json"
        if not results_json.exists():
            return None
        try:
            with open(results_json) as f:
                data = json.load(f)
            cases = data.get("results", [])
            if not cases:
                return None

            def _normalize_speedup(r: dict) -> float:
                v = r.get("speedup")
                try:
                    return float(v) if v is not None else float("inf")
                except (TypeError, ValueError):
                    return float("inf")

            # Only consider PASS cases with a valid speedup > 0 to avoid
            # selecting FAIL/ERROR entries that record speedup=0.0.
            pass_cases = [
                r for r in cases
                if str(r.get("status", "")).upper() == "PASS" and _normalize_speedup(r) > 0
            ]
            candidates = pass_cases if pass_cases else cases
            worst = min(candidates, key=_normalize_speedup)
            name = worst.get("case_name")
            speedup_val = worst.get("speedup")
            if speedup_val is not None:
                try:
                    log_info(f"  Worst case from results.json: {name} (speedup={float(speedup_val):.2f}x)")
                except (TypeError, ValueError):
                    log_info(f"  Worst case from results.json: {name} (speedup={speedup_val})")
            else:
                log_info(f"  Worst case from results.json: {name} (speedup=?)")
            return name
        except Exception as e:
            log_warning(f"  Failed to parse results.json: {e}")
            return None


class FileCopier:
    """文件拷贝器 - 负责实际的文件拷贝操作"""

    def __init__(self, config: InitConfig):
        self.config = config

    def should_exclude(self, file_name: str) -> bool:
        """检查文件是否应该被排除"""
        return any(
            fnmatch.fnmatch(file_name, pattern)
            for pattern in self.config.exclude_patterns
        )

    def copy_tree(self, src: Path, dst: Path) -> CopyStats:
        """
        递归拷贝目录树

        Args:
            src: 源目录
            dst: 目标目录

        Returns:
            拷贝统计信息
        """
        stats = CopyStats()

        if not src.exists():
            return stats

        # Dry-run 模式：只记录，不实际拷贝
        if self.config.dry_run:
            for _root, _, files in os.walk(src):
                for file_name in files:
                    if self.should_exclude(file_name):
                        stats.skipped_files += 1
                    else:
                        stats.total_files += 1
            return stats

        # 实际拷贝
        dst.mkdir(parents=True, exist_ok=True)

        for root, _, files in os.walk(src):
            rel_path = Path(root).relative_to(src)
            target_dir = dst / rel_path
            target_dir.mkdir(parents=True, exist_ok=True)

            for file_name in files:
                stats.total_files += 1

                # 检查排除规则
                if self.should_exclude(file_name):
                    stats.skipped_files += 1
                    continue

                src_file = Path(root) / file_name
                dst_file = target_dir / file_name

                # 检查是否覆盖
                if dst_file.exists() and not self.config.overwrite_existing:
                    stats.skipped_files += 1
                    continue

                try:
                    shutil.copy2(src_file, dst_file)
                    stats.copied_files += 1
                    stats.total_size_bytes += src_file.stat().st_size

                    # 验证拷贝（可选）
                    if self.config.verify_copy:
                        if src_file.stat().st_size != dst_file.stat().st_size:
                            log_warning(f"Size mismatch: {src_file} -> {dst_file}")
                            stats.failed_files += 1
                            stats.copied_files -= 1

                except Exception as exc:
                    stats.failed_files += 1
                    log_error(f"Failed to copy {src_file}: {exc}")

        return stats

    def copy_file(self, src: Path, dst: Path) -> bool:
        """
        拷贝单个文件

        Returns:
            是否成功
        """
        if not src.exists():
            return False

        if self.config.dry_run:
            log_info(f"[DRY-RUN] Would copy: {src} -> {dst}")
            return True

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
        except Exception as exc:
            log_error(f"Failed to copy {src}: {exc}")
            return False


class WorkspaceValidator:
    """工作空间验证器 - 验证拷贝结果"""

    def __init__(self, pm: PathManager):
        self.pm = pm

    def validate_operator(self, op_name: str) -> Dict[str, bool]:
        """
        验证算子目录的完整性

        Returns:
            {"code_exists": bool, "profiling_exists": bool, ...}
        """
        input_dir = self.pm.get_input_dir(op_name)
        code_dir = self.pm.get_input_code_dir(op_name)
        profiling_dir = self.pm.get_input_profiling_dir(op_name)

        return {
            "input_dir_exists": input_dir.exists(),
            "code_dir_exists": code_dir.exists(),
            "op_host_exists": (code_dir / "op_host").exists(),
            "op_kernel_exists": (code_dir / "op_kernel").exists(),
            "profiling_dir_exists": profiling_dir.exists(),
            "op_summary_exists": (profiling_dir / "op_summary.csv").exists(),
        }


class WorkspaceInitializer:
    """工作空间初始化器 - 协调整个初始化流程"""

    def __init__(self, pm: PathManager, config: InitConfig):
        self.pm = pm
        self.config = config
        self.detector = SourceDetector(pm)
        self.copier = FileCopier(config)
        self.validator = WorkspaceValidator(pm)

    def process_operator(self, op_dir: Path) -> OperatorResult:
        """
        处理单个算子

        Args:
            op_dir: CAKE2/output/{op}/ 目录

        Returns:
            处理结果
        """
        op_name = op_dir.name
        result = OperatorResult(op_name=op_name, success=False, source_dir=op_dir)

        log_info(f"Processing operator: {op_name}")

        # 1. 检测源文件
        code_dirs = self.detector.find_code_dirs(op_dir)
        profiling_csv = self.detector.find_best_profiling_csv(op_dir)

        # 检查必需的源文件
        if not code_dirs["op_host"] and not code_dirs["op_kernel"]:
            result.errors.append("No op_host or op_kernel found")
            log_error(f"[{op_name}] No code directories found")
            return result

        # 2. 准备目标目录
        target_dir = self.pm.get_input_dir(op_name)
        result.target_dir = target_dir

        if not self.config.dry_run:
            self.pm.ensure_dir_exists(target_dir)

        # 3. 拷贝代码（根据 copy_mode）
        if self.config.copy_mode in [CopyMode.FULL, CopyMode.CODE_ONLY]:
            code_dst = self.pm.get_input_code_dir(op_name)

            for sub_name, src in code_dirs.items():
                if src is None:
                    result.warnings.append(f"{sub_name} not found")
                    log_warning(f"[{op_name}] {sub_name} not found")
                    continue

                dst = code_dst / sub_name
                log_info(f"  Copying {sub_name}: {src} -> {dst}")

                stats = self.copier.copy_tree(src, dst)
                result.stats.merge(stats)

                if stats.failed_files > 0:
                    result.errors.append(f"{sub_name}: {stats.failed_files} files failed")

        # 4. 拷贝 profiling（根据 copy_mode）
        if self.config.copy_mode in [CopyMode.FULL, CopyMode.PROFILING_ONLY]:
            if profiling_csv:
                profiling_dst = self.pm.get_input_profiling_dir(op_name)
                dst_file = profiling_dst / "op_summary.csv"

                log_info(f"  Copying profiling: {profiling_csv} -> {dst_file}")

                if self.copier.copy_file(profiling_csv, dst_file):
                    result.stats.copied_files += 1
                    result.stats.total_files += 1
                else:
                    result.errors.append("Failed to copy op_summary.csv")
            else:
                result.warnings.append("No op_summary.csv found")
                log_warning(f"[{op_name}] No profiling CSV found")

        # 5. 计算代码哈希（可选）
        if self.config.compute_hash and not self.config.dry_run:
            code_dir = self.pm.get_input_code_dir(op_name)
            if code_dir.exists():
                result.code_hash = compute_dir_hash(code_dir)

        # 6. 验证结果
        if not self.config.dry_run:
            validation = self.validator.validate_operator(op_name)

            # 检查关键文件
            if not validation["code_dir_exists"]:
                result.errors.append("Code directory not created")

            if self.config.copy_mode == CopyMode.FULL:
                if not validation["op_summary_exists"]:
                    result.warnings.append("Profiling data missing")

        # 7. 判断成功/失败
        result.success = (
            result.stats.copied_files > 0 and
            len(result.errors) == 0
        )

        return result

    def run(self, filter_op: Optional[str] = None) -> Dict[str, OperatorResult]:
        """
        运行初始化流程

        Args:
            filter_op: 可选，仅处理指定算子

        Returns:
            {op_name: OperatorResult}
        """
        results = {}

        # 1. 检测 output 目录
        try:
            output_dir = self.detector.detect_output_dir()
            log_success(f"Output directory: {output_dir}")
        except FileNotFoundError as e:
            log_error(str(e))
            return results

        # 2. 列出算子
        try:
            op_dirs = self.detector.list_operators(output_dir, filter_op)
            log_info(f"Found {len(op_dirs)} operator(s)")
        except FileNotFoundError as e:
            log_error(str(e))
            return results

        if not op_dirs:
            log_warning("No operators found")
            return results

        # 3. 处理每个算子
        for op_dir in op_dirs:
            try:
                result = self.process_operator(op_dir)
                results[result.op_name] = result
            except Exception as exc:
                log_error(f"Unexpected error processing {op_dir.name}: {exc}")
                results[op_dir.name] = OperatorResult(
                    op_name=op_dir.name,
                    success=False,
                    errors=[str(exc)]
                )

        return results


# ========== 报告生成 ==========


class Reporter:
    """报告生成器"""

    @staticmethod
    def print_summary(results: Dict[str, OperatorResult], config: InitConfig):
        """打印汇总报告"""
        total_ops = len(results)
        success_ops = sum(1 for r in results.values() if r.success)
        failed_ops = total_ops - success_ops

        total_stats = CopyStats()
        for result in results.values():
            total_stats.merge(result.stats)

        logger.info("\n" + "=" * 60)
        logger.info("📊 INITIALIZATION SUMMARY")
        logger.info("=" * 60)

        if config.dry_run:
            logger.info("🔍 Mode: DRY-RUN (no files actually copied)")
        else:
            logger.info(f"✅ Mode: {config.copy_mode.value}")

        logger.info(f"\nOperators: {total_ops} total, {success_ops} success, {failed_ops} failed")
        logger.info(
            f"Files: {total_stats.copied_files} copied, {total_stats.skipped_files} skipped, {total_stats.failed_files} failed")

        if total_stats.total_size_bytes > 0:
            size_mb = total_stats.total_size_bytes / (1024 * 1024)
            logger.info(f"Size: {size_mb:.2f} MB")

        # 详细结果
        if results:
            logger.info("\n" + "-" * 60)
            logger.info("📋 OPERATOR DETAILS")
            logger.info("-" * 60)

            for op_name, result in sorted(results.items()):
                status = "✅" if result.success else "❌"
                logger.info(f"\n{status} {op_name}")

                if result.target_dir:
                    logger.info(f"   Target: {result.target_dir}")

                if result.code_hash:
                    logger.info(f"   Hash: {result.code_hash}")

                logger.info(
                    f"   Files: {result.stats.copied_files} copied, {result.stats.skipped_files} skipped, {result.stats.failed_files} failed")

                if result.warnings:
                    for warning in result.warnings:
                        logger.info(f"   ⚠️  {warning}")

                if result.errors:
                    for error in result.errors:
                        logger.info(f"   ❌ {error}")

        logger.info("\n" + "=" * 60)

    @staticmethod
    def save_report(results: Dict[str, OperatorResult], output_file: Path):
        """保存报告到 JSON 文件"""
        report = {
            "timestamp": Path(output_file).stat().st_mtime if output_file.exists() else 0,
            "operators": {}
        }

        for op_name, result in results.items():
            report["operators"][op_name] = {
                "success": result.success,
                "stats": {
                    "copied": result.stats.copied_files,
                    "skipped": result.stats.skipped_files,
                    "failed": result.stats.failed_files,
                },
                "code_hash": result.code_hash,
                "errors": result.errors,
                "warnings": result.warnings,
                "source_dir": str(result.source_dir) if result.source_dir else None,
                "target_dir": str(result.target_dir) if result.target_dir else None,
            }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        log_success(f"Report saved: {output_file}")


# ========== CLI ==========


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Initialize workspace inputs from CAKE2 output artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize all operators
  python3 init_workspace.py

  # Initialize specific operator
  python3 init_workspace.py --op fastgelu

  # Dry-run (preview only)
  python3 init_workspace.py --dry-run

  # Copy only code (skip profiling)
  python3 init_workspace.py --code-only

  # Custom CAKE2 root
  python3 init_workspace.py --root /path/to/CAKE2

  # Overwrite existing files
  python3 init_workspace.py --overwrite
        """
    )

    parser.add_argument(
        "--root",
        type=str,
        default="",
        help="CAKE2 root path (auto-detected if omitted)"
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default="",
        help="Workspace directory override (absolute or relative to CAKE2 root). Useful for per-branch isolation."
    )

    parser.add_argument(
        "--op",
        type=str,
        default="",
        help="Process only one operator (e.g., fastgelu)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually copying files"
    )

    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Copy only code (skip profiling data)"
    )

    parser.add_argument(
        "--profiling-only",
        action="store_true",
        help="Copy only profiling data (skip code)"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: skip)"
    )

    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip copy verification (faster but less safe)"
    )

    parser.add_argument(
        "--compute-hash",
        action="store_true",
        help="Compute code hash for change detection"
    )

    parser.add_argument(
        "--save-report",
        type=str,
        default="",
        help="Save report to JSON file (e.g., report.json)"
    )

    return parser.parse_args()


def detect_skill_root(explicit_root: Optional[str] = None) -> Path:
    """
    检测 code-performance-advisor skill 根目录

    Args:
        explicit_root: 可选，CAKE2 根目录（如果提供，则推导 skill 根目录）

    Returns:
        Skill 根目录路径
    """
    if explicit_root:
        # 如果用户提供了 CAKE2 根目录，推导 skill 根目录
        cake2_root = Path(explicit_root).expanduser().resolve()
        skill_root = cake2_root / ".claude" / "skills" / "code-performance-advisor"

        if skill_root.exists():
            return skill_root
        else:
            raise FileNotFoundError(f"Skill directory not found: {skill_root}")

    # 自动检测：从当前脚本位置向上查找
    current = Path(__file__).resolve()

    # 从脚本所在目录向上查找（预期在 scripts/analysis_engine/）
    for parent in [current.parent.parent.parent] + list(current.parents):
        if (parent / "assets" / "configs" / "paths.yaml").exists():
            return parent

    raise FileNotFoundError(
        "Unable to detect skill root. Please pass --root /path/to/CAKE2"
    )


def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = parse_args()

    # 1. 检测 skill 根目录
    try:
        skill_root = detect_skill_root(args.root or None)
        log_success(f"Skill root: {skill_root}")
    except FileNotFoundError as e:
        log_error(str(e))
        sys.exit(1)

    # 2. 初始化 PathManager
    pm = PathManager(skill_root, workspace_override=Path(args.workspace) if args.workspace else None)

    # 验证外部路径
    validation = pm.validate_external_paths()
    if not validation["cake2_root_exists"]:
        log_error(f"CAKE2 root not found: {validation['cake2_root_path']}")
        log_info("Please check assets/configs/paths.yaml or use --root option")
        sys.exit(1)

    log_success(f"CAKE2 root: {pm.get_cake2_root()}")
    log_success(f"Workspace: {pm.get_workspace_dir()}")

    # 3. 构建配置
    copy_mode = CopyMode.FULL
    if args.code_only:
        copy_mode = CopyMode.CODE_ONLY
    elif args.profiling_only:
        copy_mode = CopyMode.PROFILING_ONLY

    config = InitConfig(
        dry_run=args.dry_run,
        copy_mode=copy_mode,
        overwrite_existing=args.overwrite,
        verify_copy=not args.no_verify,
        compute_hash=args.compute_hash,
    )

    # 4. 运行初始化
    initializer = WorkspaceInitializer(pm, config)
    results = initializer.run(filter_op=args.op or None)

    # 5. 打印报告
    Reporter.print_summary(results, config)

    # 6. 保存报告（可选）
    if args.save_report:
        report_file = Path(args.save_report)
        Reporter.save_report(results, report_file)

    # 7. 返回退出码
    failed_count = sum(1 for r in results.values() if not r.success)
    if failed_count > 0:
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
