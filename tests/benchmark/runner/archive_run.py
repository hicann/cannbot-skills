#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""归档脚本：将评测生成的算子交付工程通过 git 分支归档到 remote。

交付格式对齐 cann-bench examples/direct_launch_example:
  {op}/                       完整可构建工程
    ├── CMakeLists.txt / setup.py / build.sh / ...   构建文件 (取自模板)
    ├── csrc/ops/{op}/        算子源码 (kernel/plugin)
    ├── cann_bench/__init__.py 注册入口
    ├── tests/{op}/           算子测试
    └── dist/*.whl            交付件

数据源为 run_eval.py 持久化的 operators/{op}/ 产物 (而非 cann-bench 树内
扫描 —— 评测在隔离 workdir 中生成代码, cann-bench examples 不含源码)。

流程:
  1. 基于 direct_launch_example 模板 + operators/{op}/ 组装交付工程并校验
  2. 在 cann-bench 中创建归档分支, 提交 eval_delivery/{op}/
  3. 推送到 origin
  4. 切回原始分支, 删除本地归档分支与暂存文件 (磁盘零残留)

用法:
  python runner/archive_run.py --name run-001
  python runner/archive_run.py --dry-run
  python runner/archive_run.py --name run-001 --force
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

BENCHMARK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPERATORS_DIR = os.path.join(BENCHMARK_ROOT, "operators")

# 归档分支内的交付根目录
ARCHIVE_ROOT = "eval_delivery"
# 交付工程模板 (cann-bench 内)
TEMPLATE_EXAMPLE = os.path.join("examples", "direct_launch_example")

# 模板自带的基线算子/测试 (非评测产物, 组装时剔除)
TEMPLATE_BASELINE = {"add", "sqrt"}

# 交付工程必需文件 (缺失 = error) / 建议文件 (缺失 = warning)
REQUIRED_PATHS = ["CMakeLists.txt", "setup.py", "build.sh",
                  "cann_bench/__init__.py"]
RECOMMENDED_DIRS = ["tests", "dist"]


def _run(cmd, cwd=None, check=True):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)


def get_cann_bench_root():
    from setup_cann_bench import ensure_cann_bench
    return ensure_cann_bench()


# ══════════════════════════════════════════════════════════════════════
#  交付工程组装 (operators/ 持久化产物 + direct_launch_example 模板)
# ══════════════════════════════════════════════════════════════════════

def find_persisted_ops(operators_dir: str = OPERATORS_DIR) -> list[str]:
    """扫描 operators/ 下含算子源码 (csrc/ops/*/) 的持久化算子。"""
    ops = []
    if not os.path.isdir(operators_dir):
        return ops
    for entry in sorted(os.listdir(operators_dir)):
        csrc_ops = os.path.join(operators_dir, entry, "csrc", "ops")
        if not os.path.isdir(csrc_ops):
            continue
        if any(os.path.isdir(os.path.join(csrc_ops, d))
               for d in os.listdir(csrc_ops)):
            ops.append(entry)
    return ops


def _overlay_persisted_dirs(proj_section: str, persisted_section: str,
                            strip_baseline: bool = False):
    """用持久化算子的某子目录 (csrc/ops / tests) 覆盖工程内对应目录。

    先剔除模板基线 (add/sqrt), 再把持久化子目录逐个覆盖拷入。
    proj_section 仅在 persisted_section 存在时按需创建 (对齐原始 tests 行为)。
    """
    if os.path.isdir(proj_section) and strip_baseline:
        for entry in os.listdir(proj_section):
            p = os.path.join(proj_section, entry)
            if os.path.isdir(p) and entry in TEMPLATE_BASELINE:
                shutil.rmtree(p)
    if not os.path.isdir(persisted_section):
        return
    os.makedirs(proj_section, exist_ok=True)
    for entry in sorted(os.listdir(persisted_section)):
        src = os.path.join(persisted_section, entry)
        dst = os.path.join(proj_section, entry)
        if not os.path.isdir(src):
            continue
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _copy_dist_whls(persisted_dist: str, proj_dist: str):
    """把持久化 dist/ 下的 .whl 拷入交付工程 dist/。"""
    if not os.path.isdir(persisted_dist):
        return
    os.makedirs(proj_dist, exist_ok=True)
    for fn in sorted(os.listdir(persisted_dist)):
        if fn.endswith(".whl"):
            shutil.copy2(os.path.join(persisted_dist, fn),
                         os.path.join(proj_dist, fn))


def stage_delivery_project(op_slug: str, operators_dir: str,
                           template_dir: str, staging_root: str) -> str:
    """组装单个算子的交付工程到 staging_root/{op_slug}/, 返回工程路径。

    模板提供构建文件 (CMakeLists/setup.py/build.sh/cmake/scripts 等,
    均为算子无关 —— csrc/ops/CMakeLists.txt 自动发现算子目录);
    operators/{op}/ 提供算子源码/注册入口/测试/whl。
    """
    proj_dir = os.path.join(staging_root, op_slug)
    if os.path.exists(proj_dir):
        shutil.rmtree(proj_dir)

    # 1) 模板骨架 (剔除构建产物与缓存; csrc/ops 与 tests 随后按算子重建)
    shutil.copytree(
        template_dir, proj_dir, symlinks=True,
        ignore=shutil.ignore_patterns(
            "build", "build_py", "dist",
            "__pycache__", "*.pyc", "*.egg-info",
            "_C.abi3.so", "CMakeCache.txt", "CMakeFiles", "Makefile",
            "cmake_install.cmake", "*.o", "*.os",
        ),
    )

    # 2) csrc/ops: 剔除模板基线算子, 替换为持久化算子源码
    if not os.path.isdir(os.path.join(proj_dir, "csrc", "ops")):
        os.makedirs(os.path.join(proj_dir, "csrc", "ops"), exist_ok=True)
    _overlay_persisted_dirs(
        os.path.join(proj_dir, "csrc", "ops"),
        os.path.join(operators_dir, op_slug, "csrc", "ops"),
        strip_baseline=True)

    # 3) cann_bench/__init__.py: 用算子持久化的注册入口覆盖模板
    persisted_init = os.path.join(operators_dir, op_slug,
                                  "cann_bench", "__init__.py")
    if os.path.isfile(persisted_init):
        shutil.copy2(persisted_init,
                     os.path.join(proj_dir, "cann_bench", "__init__.py"))

    # 4) tests: 剔除模板基线测试, 替换为持久化算子测试
    _overlay_persisted_dirs(
        os.path.join(proj_dir, "tests"),
        os.path.join(operators_dir, op_slug, "tests"),
        strip_baseline=True)

    # 5) dist: 交付件 .whl
    _copy_dist_whls(os.path.join(operators_dir, op_slug, "dist"),
                    os.path.join(proj_dir, "dist"))

    return proj_dir


def validate_delivery_project(proj_dir: str, op_slug: str
                              ) -> tuple[list[str], list[str]]:
    """按 direct_launch_example 交付格式校验, 返回 (errors, warnings)。"""
    errors, warnings = [], []

    for rel in REQUIRED_PATHS:
        if not os.path.isfile(os.path.join(proj_dir, rel)):
            errors.append(f"缺少必需文件: {rel}")

    csrc_ops = os.path.join(proj_dir, "csrc", "ops")
    op_dirs = [d for d in os.listdir(csrc_ops)
               if os.path.isdir(os.path.join(csrc_ops, d))] \
        if os.path.isdir(csrc_ops) else []
    if not op_dirs:
        errors.append("缺少算子源码目录: csrc/ops/{op}/")
    else:
        for d in op_dirs:
            has_src = any(fn.endswith((".cpp", ".h", ".hpp"))
                          for _, _, files in os.walk(os.path.join(csrc_ops, d))
                          for fn in files)
            if not has_src:
                errors.append(f"算子源码目录无源文件: csrc/ops/{d}/")

    tests_dir = os.path.join(proj_dir, "tests")
    test_dirs = [d for d in os.listdir(tests_dir)
                 if os.path.isdir(os.path.join(tests_dir, d))] \
        if os.path.isdir(tests_dir) else []
    if not test_dirs:
        warnings.append("缺少算子测试: tests/{op}/")

    # 模板基线 API 残留: init 仍引用已被剔除的 add/sqrt
    # (算子 __init__.py 未持久化时保留了模板注册入口, 调用将失败)
    init_path = os.path.join(proj_dir, "cann_bench", "__init__.py")
    if os.path.isfile(init_path):
        with open(init_path, encoding="utf-8") as f:
            init_src = f.read()
        for baseline in TEMPLATE_BASELINE:
            if (f"torch.ops.cann_bench.{baseline}" in init_src
                    and not os.path.isdir(os.path.join(csrc_ops, baseline))):
                warnings.append(
                    f"注册入口残留模板基线 API '{baseline}' "
                    f"(csrc/ops/{baseline}/ 已剔除; 算子 __init__.py 未持久化?)")

    dist_dir = os.path.join(proj_dir, "dist")
    if not os.path.isdir(dist_dir) or not any(
            fn.endswith(".whl") for fn in os.listdir(dist_dir)):
        warnings.append("缺少交付件: dist/*.whl")

    return errors, warnings


# ══════════════════════════════════════════════════════════════════════
#  git 归档
# ══════════════════════════════════════════════════════════════════════

def get_current_branch(cann_bench_root: str) -> str:
    result = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cann_bench_root)
    return result.stdout.strip()


def archive_submodule(cann_bench_root: str, branch_name: str, remote: str,
                      staging_root: str, op_slugs: list[str],
                      dry_run: bool = False) -> bool:
    original_branch = get_current_branch(cann_bench_root)
    archive_rel = ARCHIVE_ROOT

    print(f"原始分支: {original_branch}")
    print(f"归档分支: {branch_name}")
    print(f"\n将归档以下交付工程到 {archive_rel}/:")
    for op in op_slugs:
        print(f"  {archive_rel}/{op}/")
    print(f"\n共 {len(op_slugs)} 个算子")

    if dry_run:
        print("\n[DRY-RUN] 不会实际执行。")
        return True

    archive_abs = os.path.join(cann_bench_root, archive_rel)
    pushed = False
    created = False  # 分支是否由本次运行创建 (避免 finally 误删已有同名分支)
    try:
        print(f"\n>>> 创建分支 {branch_name} ...")
        _run(["git", "checkout", "-b", branch_name], cwd=cann_bench_root)
        created = True

        print(">>> 复制交付工程到工作区 ...")
        if os.path.exists(archive_abs):
            shutil.rmtree(archive_abs)
        shutil.copytree(os.path.join(staging_root, ARCHIVE_ROOT), archive_abs)

        print(">>> 添加文件到暂存区 ...")
        _run(["git", "add", "-f", archive_rel], cwd=cann_bench_root)

        result = _run(["git", "status", "--short"], cwd=cann_bench_root,
                      check=False)
        print(">>> 暂存区状态 (前 20 行):")
        print("\n".join(result.stdout.splitlines()[:20]))

        print(">>> 提交 ...")
        commit_msg = (f"eval({branch_name}): {len(op_slugs)} operators delivered "
                      f"({', '.join(op_slugs)})")
        _run(["git", "commit", "-m", commit_msg], cwd=cann_bench_root)

        print(f">>> 推送到 {remote} ...")
        _run(["git", "push", remote, branch_name], cwd=cann_bench_root)
        pushed = True
    finally:
        print(f">>> 切回 {original_branch} ...")
        _run(["git", "checkout", original_branch], cwd=cann_bench_root,
             check=False)
        if created:
            print(f">>> 删除本地分支 {branch_name} ...")
            _run(["git", "branch", "-D", branch_name], cwd=cann_bench_root,
                 check=False)
        # 仅清理 original_branch 未跟踪的 eval_delivery/ 残留: 若该目录受跟踪,
        # git checkout 已将其还原到 HEAD 状态, 不能 rmtree (会误删受跟踪文件,
        # 导致工作区出现 "deleted" 改动)
        if os.path.isdir(archive_abs):
            tracked = _run(["git", "ls-files", archive_rel],
                           cwd=cann_bench_root, check=False).stdout.strip()
            if not tracked:
                shutil.rmtree(archive_abs, ignore_errors=True)
            else:
                _run(["git", "restore", "--", archive_rel],
                     cwd=cann_bench_root, check=False)

    if pushed:
        print(f"\n归档完成! 分支 {branch_name} 已推送到 {remote}, 本地已删除。")
    return pushed


def archive_results(branch_name: str, dry_run: bool = False):
    results_dir = os.path.join(BENCHMARK_ROOT, "results")
    if not os.path.isdir(results_dir) or not os.listdir(results_dir):
        print("\nresults/ 为空或不存在, 跳过归档。")
        return

    print(f"\n>>> results/ 目录包含评测结果, 请手动备份到: {results_dir}")


def _auto_detect_name(cann_bench_root: str) -> str:
    """自动检测下一个可用的归档名称 eval-run-<NNN>。"""
    result = _run(["git", "branch", "-a"], cwd=cann_bench_root, check=False)
    existing = [
        line.strip()
        for line in result.stdout.split("\n")
        if "eval-run-" in line
    ]
    seq = 1
    while any(f"eval-run-{seq:03d}" in b for b in existing):
        seq += 1
    return f"eval-run-{seq:03d}"


def _stage_one_op(op_slug: str, operators_dir: str, template_dir: str,
                  staging_delivery: str, force: bool) -> tuple[str | None, bool]:
    """组装+校验单个算子; 返回 (归档名 或 None, 是否有 error)。"""
    proj_dir = stage_delivery_project(
        op_slug, operators_dir, template_dir, staging_delivery)
    errors, warnings = validate_delivery_project(proj_dir, op_slug)
    status = "OK" if not errors else "ERROR"
    print(f"\n[{status}] {op_slug}/")
    for w in warnings:
        print(f"  [WARN] {w}")
    for e in errors:
        print(f"  [ERROR] {e}")
    if not errors:
        return op_slug, False
    if not force:
        print(f"  [SKIP] 校验未通过, 跳过 (--force 可强制归档)")
        shutil.rmtree(proj_dir)
        return None, True
    return op_slug, True


def _confirm_archive(force: bool, dry_run: bool, yes: bool = False) -> bool:
    """确认归档; 返回是否继续。

    --dry-run / --yes / --force 任一为真均跳过交互确认;
    非交互环境 (无 tty) 且未显式指定上述标志时, 直接拒绝而非等待 input(),
    避免 CI 中 EOFError 中断 (CI 想单纯 dry-run 只需 --dry-run, 无需 --yes)。
    """
    if dry_run or yes or force:
        return True
    if not sys.stdin.isatty():
        print("[ERROR] 非交互环境需 --yes / --force / --dry-run 显式跳过确认。")
        return False
    resp = input("确认归档? (y/N): ")
    if resp.lower() in ("y", "yes"):
        return True
    print("已取消。")
    return False


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="归档评测生成物 (operators/ 持久化产物) 到 git 远程分支")
    parser.add_argument("--name", default=None,
                        help="归档名称 (默认: eval-run-<auto> 自动检测序号)")
    parser.add_argument("--remote", default="origin",
                        help="远程仓库名称 (默认: origin)")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式, 不实际执行 (非交互环境可单独使用)")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="跳过确认提示 (不绕过校验; 与 --force 区分)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="跳过确认提示; 交付工程校验 error 时仍归档")
    return parser


def main():
    args = _build_arg_parser().parse_args()

    try:
        cann_bench_root = get_cann_bench_root()
    except RuntimeError as e:  # 钉版本校验 fail-fast 报警
        print(str(e))
        return 1

    if args.name is None:
        args.name = _auto_detect_name(cann_bench_root)

    op_slugs = find_persisted_ops()
    if not op_slugs:
        print("没有检测到 operators/ 持久化产物 (csrc/ops/*/ 算子源码)。")
        print("请先运行评测 (run_eval.py 会将算子产物持久化到 operators/)。")
        return 0

    template_dir = os.path.join(cann_bench_root, TEMPLATE_EXAMPLE)
    if not os.path.isdir(template_dir):
        print(f"错误: 交付模板不存在: {template_dir}")
        return 1

    print("=" * 60)
    print(f"检测到 {len(op_slugs)} 个持久化算子, 组装交付工程 ...")
    print("=" * 60)

    staging_root = tempfile.mkdtemp(prefix="archive-staging-")
    try:
        staging_delivery = os.path.join(staging_root, ARCHIVE_ROOT)
        os.makedirs(staging_delivery, exist_ok=True)

        archived_ops = []
        has_errors = False
        for op_slug in op_slugs:
            kept, had_error = _stage_one_op(
                op_slug, OPERATORS_DIR, template_dir, staging_delivery, args.force)
            if kept:
                archived_ops.append(kept)
            if had_error:
                has_errors = True

        if not archived_ops:
            print("\n没有通过校验的交付工程, 无需归档。")
            return 1 if has_errors else 0

        print("\n" + "=" * 60)
        print(f"交付工程: {len(archived_ops)} 个算子 → {ARCHIVE_ROOT}/")
        print("=" * 60)

        if not _confirm_archive(args.force, args.dry_run, getattr(args, "yes", False)):
            return 0

        success = archive_submodule(cann_bench_root, args.name, args.remote,
                                    staging_root, archived_ops,
                                    dry_run=args.dry_run)
        if success:
            archive_results(args.name, dry_run=args.dry_run)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
