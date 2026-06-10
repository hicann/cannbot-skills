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
"""
CANN 算子仓库自动检测工具（动态发现版本）

功能：
1. 动态检测 ops-* 仓库（无需硬编码仓库列表）
2. 向上遍历父目录查找 ops-* 仓库根目录
3. 支持任意嵌套深度
4. 自动 clone 仓库到当前目录（如果找不到）
5. 自动 checkout master 分支并 pull 最新代码

使用方式：
    python repo_detector.py                    # 自动检测
    python repo_detector.py --repo ops-math    # 指定仓库
    from repo_detector import detect_ops_repo, get_repo_root
    repo_name, repo_path = detect_ops_repo()

检测规则（优先级从高到低）：
1. 当前目录包含 ops-* 子目录 → 使用当前目录下的仓库
2. 当前目录是 ops-* 仓库根目录 → 直接使用
3. 当前目录在 ops-* 仓库内部 → 向上查找父目录的 ops-*
4. 以上都不满足 → clone 到当前目录

核心改进：
- 使用动态发现替代硬编码仓库列表
- 支持任意 ops-* 仓库名（无需预先定义）
- GitCode URL 自动推断
"""

import os
import shutil
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

try:
    from config_loader import (
        get_repo_pattern, get_repo_markers, get_min_markers,
        get_excluded_dirs, build_clone_url
    )
    REPO_PATTERN = get_repo_pattern()
    OPS_REPO_MARKERS = get_repo_markers()
    MIN_MARKERS_REQUIRED = get_min_markers()
    EXCLUDED_CONFIG_DIRS = get_excluded_dirs()
    USE_CONFIG = True
except ImportError:
    REPO_PATTERN = "ops-"
    OPS_REPO_MARKERS = ['CMakeLists.txt', 'docs', 'op_host', 'op_kernel', 'common', 'cmake', 'examples']
    MIN_MARKERS_REQUIRED = 2
    EXCLUDED_CONFIG_DIRS = [
        'tests', 'docs', 'common', 'cmake', 'examples',
        'op_host', 'op_kernel', 'op_api', 'op_graph',
        '.git', '.github', '.opencode', 'build', 'framework'
    ]
    GITCODE_URL_TEMPLATE = "https://gitcode.com/cann/{repo}.git"
    USE_CONFIG = False

SCAN_EXCLUDE_DIRS = ['.opencode', '.git', '.github', 'build', '.idea', '.vscode']
SCAN_PRODUCT_DIRS = ['reports', 'tmp', 'legacy']


def is_ops_repo_root(path: Path, repo_name: str = None) -> bool:
    """判断给定路径是否是 ops-* 仓库的根目录"""
    if not path.exists() or not path.is_dir():
        return False
    
    if repo_name and path.name != repo_name:
        return False
    
    if not path.name.startswith(REPO_PATTERN):
        return False
    
    markers_found = sum(1 for marker in OPS_REPO_MARKERS if (path / marker).exists())
    return markers_found >= MIN_MARKERS_REQUIRED


def find_ops_repos_in_dir(path: Path) -> List[Tuple[str, Path]]:
    """在给定目录下查找所有 ops-* 子目录（动态发现）"""
    found = []
    if not path.exists() or not path.is_dir():
        return found
    
    for item in path.iterdir():
        if item.is_dir() and item.name.startswith(REPO_PATTERN):
            if is_ops_repo_root(item, item.name):
                found.append((item.name, item))
    
    return found


def is_inside_ops_repo(path: Path) -> bool:
    """判断路径是否在 ops-* 仓库内部（通过查找 .git 目录）"""
    current = path
    while current != current.parent:
        git_dir = current / '.git'
        if git_dir.exists() and current.name.startswith(REPO_PATTERN):
            return True
        parent_ops = current.parent
        if parent_ops.name.startswith(REPO_PATTERN) and (parent_ops / '.git').exists():
            return True
        current = current.parent
    return False


def get_ops_repo_parent(path: Path) -> Optional[Path]:
    """获取路径所在的 ops-* 仓库父目录"""
    current = path
    while current != current.parent:
        if current.name.startswith(REPO_PATTERN) and (current / '.git').exists():
            return current
        parent = current.parent
        if parent.name.startswith(REPO_PATTERN) and (parent / '.git').exists():
            return parent
        current = current.parent
    return None


def clone_repo_to_current_dir(repo_type: str, target_dir: Path = None) -> Tuple[Path, str]:
    """
    克隆仓库到当前目录
    
    Args:
        repo_type: 仓库名（ops-math, ops-nn, ops-transformer, ops-cv 或任意 ops-*）
        target_dir: 目标目录（默认为当前目录）
    
    Returns:
        Tuple[Path, str]: (clone_path, method)
    """
    if target_dir is None:
        target_dir = Path.cwd()
    
    if not repo_type.startswith(REPO_PATTERN):
        raise ValueError(f"无效的仓库名: {repo_type}（应以 {REPO_PATTERN} 开头）")
    
    clone_url = build_clone_url(repo_type) if USE_CONFIG else GITCODE_URL_TEMPLATE.format(repo=repo_type)
    clone_path = target_dir / repo_type
    
    print(f"正在克隆 {repo_type} 仓库到 {clone_path}...")
    
    try:
        git_path = shutil.which('git')
        if git_path is None:
            raise ValueError("无法找到 git 命令，请确保 git 已安装")
        
        subprocess.run(
            [git_path, 'clone', clone_url, str(clone_path)],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"克隆完成: {clone_path}")
        return clone_path, 'cloned_to_current_dir'
    except subprocess.CalledProcessError as e:
        raise ValueError(f"克隆失败: {e.stderr}") from e


def update_repo_to_master(repo_path: Path) -> Tuple[str, str]:
    """
    更新仓库到 master 分支最新代码
    
    Args:
        repo_path: 仓库路径
    
    Returns:
        Tuple[str, str]: (branch, status)
    """
    if not (repo_path / '.git').exists():
        return 'unknown', 'not_a_git_repo'
    
    git_path = shutil.which('git')
    if git_path is None:
        return 'unknown', 'git_not_found'
    
    try:
        # 获取默认分支（master 或 main）
        result = subprocess.run(
            [git_path, 'symbolic-ref', 'refs/remotes/origin/HEAD'],
            cwd=str(repo_path),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # 解析 refs/remotes/origin/HEAD -> refs/remotes/origin/master
            default_branch = result.stdout.strip().split('/')[-1]
        else:
            # 尝试 master，如果失败则用 main
            default_branch = 'master'
        
        # Checkout 到默认分支
        subprocess.run(
            [git_path, 'checkout', default_branch],
            cwd=str(repo_path),
            check=True,
            capture_output=True,
            text=True
        )
        
        # Pull 最新代码
        result = subprocess.run(
            [git_path, 'pull', 'origin', default_branch],
            cwd=str(repo_path),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            status = 'updated'
        else:
            # 可能已经有最新代码，或有本地修改冲突
            status = 'pull_failed_or_up_to_date'
        
        return default_branch, status
        
    except subprocess.CalledProcessError as e:
        return 'unknown', f'error: {e.stderr}'


def detect_ops_repo(start_path: Path = None, target_repo: str = None, 
                    auto_clone: bool = True) -> Tuple[Optional[str], Optional[Path]]:
    """
    自动检测 ops-* 仓库
    
    检测逻辑（优先级从高到低）：
    1. 当前目录包含 ops-* 子目录 → 使用当前目录下的仓库
    2. 当前目录是 ops-* 仓库根目录 → 直接使用
    3. 当前目录在 ops-* 仓库内部（向上查找 .git）→ 使用该 ops-* 仓库
    4. 以上都不满足 → clone 到当前目录
    
    Args:
        start_path: 起始检测路径（默认为当前目录）
        target_repo: 目标仓库名（可选）
        auto_clone: 是否自动克隆（默认 True）
    
    Returns:
        Tuple[Optional[str], Optional[Path]]: (repo_name, repo_path) 或 (None, None)
    """
    if start_path is None:
        start_path = Path.cwd()
    
    start_path = start_path.resolve()
    
    # 优先级 1: 当前目录包含 ops-* 子目录
    found_in_current = find_ops_repos_in_dir(start_path)
    if found_in_current:
        if target_repo:
            for repo_name, repo_path in found_in_current:
                if repo_name == target_repo:
                    return repo_name, repo_path
        else:
            return found_in_current[0]
    
    # 优先级 2: 当前目录是 ops-* 仓库根目录
    if is_ops_repo_root(start_path, target_repo):
        return start_path.name, start_path
    
    # 优先级 3: 当前目录在 ops-* 仓库内部（必须通过 .git 验证）
    ops_parent = get_ops_repo_parent(start_path)
    if ops_parent:
        if target_repo and ops_parent.name == target_repo:
            return ops_parent.name, ops_parent
        elif not target_repo:
            return ops_parent.name, ops_parent
    
    # 优先级 4: 自动 clone（如果启用）
    if auto_clone and target_repo:
        try:
            clone_path, method = clone_repo_to_current_dir(target_repo, start_path)
            return target_repo, clone_path
        except ValueError as e:
            print(f"克隆失败: {e}")
            return None, None
    
    return None, None


def get_repo_root(repo_type: str, repo_root_arg: str = None, 
                  auto_detect: bool = True, auto_clone: bool = True,
                  update_to_master: bool = True) -> Tuple[Path, str]:
    """
    获取仓库根目录（集成自动检测和更新功能）
    
    这是各扫描脚本应使用的统一入口函数。
    
    Args:
        repo_type: 仓库类型（ops-math, ops-nn, ops-transformer, ops-cv）
        repo_root_arg: 用户指定的仓库根目录
        auto_detect: 是否启用自动检测
        auto_clone: 是否自动克隆（找不到时）
        update_to_master: 是否更新到 master 分支最新代码
    
    Returns:
        Tuple[Path, str]: (repo_root_path, detection_method)
    
    Raises:
        ValueError: 无法确定仓库路径时抛出
    """
    if repo_root_arg:
        repo_root = Path(repo_root_arg)
        if not repo_root.exists():
            raise ValueError(f"指定的仓库路径不存在: {repo_root_arg}")
        repo_root = repo_root.resolve()
        
        # 更新到 master
        if update_to_master:
            branch, status = update_repo_to_master(repo_root)
            logger.info(f"仓库分支: {branch}, 状态: {status}")
        
        return repo_root, 'user_specified'
    
    if auto_detect:
        detected_name, detected_path = detect_ops_repo(
            target_repo=repo_type, 
            auto_clone=auto_clone
        )
        
        if detected_path:
            # 更新到 master
            if update_to_master:
                branch, status = update_repo_to_master(detected_path)
                logger.info(f"仓库分支: {branch}, 状态: {status}")
            
            return detected_path, f'auto_detected ({detected_name})'
    
    raise ValueError(
        f"无法确定 {repo_type} 仓库路径。\n"
        f"请通过以下方式之一指定仓库路径:\n"
        f"  1. 使用 --repo-root 参数指定绝对路径\n"
        f"  2. 确保当前目录包含 {repo_type} 子目录\n"
        f"  3. 在 {repo_type} 仓库目录或其子目录中运行命令\n"
        f"  4. 启用自动克隆（默认启用）"
    )


def is_ops_qa_suite_dir(path: Path) -> bool:
    """判断给定路径是否是 ops-qa-suite 目录（有 .opencode 配置）
    
    支持多种命名格式（兼容历史命名）：
    - ops-qa-suite（新名称）
    - repository_scan（历史命名-下划线）
    - repository-scan（历史命名-连接符）
    """
    if not path.exists() or not path.is_dir():
        return False
    
    # 支持多种命名格式（兼容历史）
    valid_names = ['ops-qa-suite', 'repository_scan', 'repository-scan']
    if path.name not in valid_names:
        return False
    
    # 检查是否有 .opencode/skills/ 目录（标识 ops-qa-suite）
    opencode_skills = path / '.opencode' / 'skills'
    return opencode_skills.exists() and opencode_skills.is_dir()


def detect_ops_qa_suite_dir(start_path: Path = None) -> Optional[Path]:
    """
    检测 ops-qa-suite 目录
    
    检测逻辑：
    1. 当前目录是 ops-qa-suite → 直接返回
    2. 当前目录包含 ops-qa-suite 子目录 → 返回子目录
    3. 向上遍历父目录查找 ops-qa-suite 目录
    4. 向上遍历查找 ops-* 仓库，检查是否有 ops-qa-suite 子目录
    
    Returns:
        Optional[Path]: ops-qa-suite 目录路径，或 None
    """
    if start_path is None:
        start_path = Path.cwd()
    
    start_path = start_path.resolve()
    
    # 支持多种命名格式（兼容历史）
    valid_names = ['ops-qa-suite', 'repository_scan', 'repository-scan']
    
    # 1. 当前目录是 ops-qa-suite
    if is_ops_qa_suite_dir(start_path):
        return start_path
    
    # 2. 当前目录包含 ops-qa-suite 等子目录
    for subdir_name in valid_names:
        scan_subdir = start_path / subdir_name
        if is_ops_qa_suite_dir(scan_subdir):
            return scan_subdir
    
    # 3. 向上遍历父目录查找
    current = start_path.parent
    while current != current.parent:
        if is_ops_qa_suite_dir(current):
            return current
        
        # 检查父目录下的子目录
        for subdir_name in valid_names:
            scan_in_parent = current / subdir_name
            if is_ops_qa_suite_dir(scan_in_parent):
                return scan_in_parent
        
        # 4. 检查 ops-* 仓库是否有子目录
        for item in current.iterdir():
            if item.is_dir() and item.name.startswith(REPO_PATTERN):
                for subdir_name in valid_names:
                    scan_in_ops = item / subdir_name
                    if is_ops_qa_suite_dir(scan_in_ops):
                        return scan_in_ops
        
        current = current.parent
    
    return None


def get_reports_output_dir(start_path: Path = None, repo_type: str = None, 
                           repo_root_arg: str = None) -> Tuple[Path, str]:
    """
    获取 reports 输出目录
    
    这是各扫描脚本应使用的统一入口函数，用于确定报告输出位置。
    
    输出目录优先级：
    1. ops-qa-suite 目录存在 → ops-qa-suite/reports/
    2. ops-* 仓库根目录 → {repo_root}/reports/
    3. 当前目录 → {cwd}/reports/
    
    Args:
        start_path: 起始检测路径（默认为当前目录）
        repo_type: 仓库类型（用于确定仓库根目录）
        repo_root_arg: 用户指定的仓库根目录（优先级最高）
    
    Returns:
        Tuple[Path, str]: (reports_dir, detection_method)
    
    Example:
        reports_dir, method = get_reports_output_dir(repo_type='ops-math')
        # 返回: (Path('.../ops-qa-suite/reports'), 'ops_qa_suite_detected')
    """
    if start_path is None:
        start_path = Path.cwd()
    
    start_path = start_path.resolve()
    
    scan_dir = detect_ops_qa_suite_dir(start_path)
    if scan_dir:
        reports_dir = scan_dir / 'reports'
        return reports_dir, f'ops_qa_suite_detected ({scan_dir})'
    
    if repo_type:
        repo_root, repo_method = get_repo_root(repo_type, repo_root_arg, update_to_master=False)
        reports_dir = repo_root / 'reports'
        return reports_dir, f'repo_root ({repo_method})'
    
    detected_name, detected_path = detect_ops_repo(start_path=start_path, auto_clone=False)
    if detected_path:
        reports_dir = detected_path / 'reports'
        return reports_dir, f'ops_repo_detected ({detected_name})'
    
    reports_dir = start_path / 'reports'
    return reports_dir, 'current_directory_fallback'


def get_detection_info(repo_type: str = None) -> dict:
    """获取当前环境的检测信息（用于调试和诊断）"""
    cwd = Path.cwd()
    result = {
        'cwd': str(cwd),
        'cwd_name': cwd.name,
        'detected_repo': None,
        'detected_path': None,
        'ops_qa_suite_dir': None,
        'reports_output_dir': None,
        'ops_repos_in_current': [],
        'ops_repos_in_parent_chain': [],
        'is_ops_repo_root': False,
        'is_inside_ops_repo': False,
    }
    
    result['is_ops_repo_root'] = is_ops_repo_root(cwd, repo_type)
    if result['is_ops_repo_root']:
        result['detected_repo'] = cwd.name
        result['detected_path'] = str(cwd)
    
    result['is_inside_ops_repo'] = is_inside_ops_repo(cwd)
    
    scan_dir = detect_ops_qa_suite_dir(cwd)
    if scan_dir:
        result['ops_qa_suite_dir'] = str(scan_dir)
    
    # 检测 reports 输出目录（捕获异常以避免中断）
    try:
        reports_dir, reports_method = get_reports_output_dir(cwd, repo_type)
        result['reports_output_dir'] = str(reports_dir)
        result['reports_detection_method'] = reports_method
    except ValueError:
        result['reports_output_dir'] = None
        result['reports_detection_method'] = 'detection_failed'
    
    found_current = find_ops_repos_in_dir(cwd)
    result['ops_repos_in_current'] = [(name, str(path)) for name, path in found_current]
    
    current = cwd.parent
    chain = []
    while current != current.parent:
        found = find_ops_repos_in_dir(current)
        if found:
            chain.append({
                'parent_path': str(current),
                'repos_found': [(name, str(path)) for name, path in found]
            })
        current = current.parent
    
    result['ops_repos_in_parent_chain'] = chain
    
    detected_name, detected_path = detect_ops_repo(target_repo=repo_type, auto_clone=False)
    if detected_path:
        result['detected_repo'] = detected_name
        result['detected_path'] = str(detected_path)
    
    return result


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='CANN 算子仓库自动检测工具（动态发现）')
    parser.add_argument('--repo', type=str,
                        help='目标仓库名（可选，支持任意 ops-* 仓库）')
    parser.add_argument('--repo-root', type=str,
                        help='仓库根目录路径（可选）')
    parser.add_argument('--reports-dir', action='store_true',
                        help='显示 reports 输出目录')
    parser.add_argument('--info', action='store_true',
                        help='显示详细的检测环境信息')
    parser.add_argument('--clone', action='store_true',
                        help='强制克隆仓库到当前目录')
    parser.add_argument('--update', action='store_true',
                        help='更新仓库到 master 分支最新代码')
    
    args = parser.parse_args()
    
    if args.info:
        info = get_detection_info(args.repo)
        print("=== 仓库检测环境信息 ===")
        print(f"当前工作目录: {info['cwd']}")
        print(f"当前目录名: {info['cwd_name']}")
        print(f"是否是 ops-* 仓库根目录: {info['is_ops_repo_root']}")
        print(f"是否在 ops-* 仓库内部: {info['is_inside_ops_repo']}")
        print()
        
        if info.get('ops_qa_suite_dir'):
            print(f"检测到 ops-qa-suite 目录: {info['ops_qa_suite_dir']}")
        else:
            print("未检测到 ops-qa-suite 目录")
        print()
        
        if info.get('reports_output_dir'):
            print(f"Reports 输出目录: {info['reports_output_dir']}")
            print(f"检测方式: {info.get('reports_detection_method', 'unknown')}")
        print()
        
        if info['ops_repos_in_current']:
            print("当前目录下找到的 ops-* 仓库:")
            for name, path in info['ops_repos_in_current']:
                print(f"  - {name}: {path}")
        else:
            print("当前目录下未找到 ops-* 仓库")
        print()
        
        if info['ops_repos_in_parent_chain']:
            print("父目录链中找到的 ops-* 仓库:")
            for item in info['ops_repos_in_parent_chain']:
                print(f"  父目录: {item['parent_path']}")
                for name, path in item['repos_found']:
                    print(f"    - {name}: {path}")
        else:
            print("父目录链中未找到 ops-* 仓库")
        print()
        
        if info['detected_repo']:
            print(f"检测结果: {info['detected_repo']}")
            print(f"仓库路径: {info['detected_path']}")
        else:
            print("检测结果: 未找到任何 ops-* 仓库")
        
        return 0
    
    if args.clone:
        if not args.repo:
            print("错误: --clone 需要指定 --repo 参数")
            return 1
        try:
            clone_path, method = clone_repo_to_current_dir(args.repo)
            print(f"克隆完成: {clone_path}")
            
            if args.update:
                branch, status = update_repo_to_master(clone_path)
                print(f"分支: {branch}, 状态: {status}")
        except ValueError as e:
            print(f"克隆失败: {e}")
            return 1
        return 0
    
    if args.update:
        if not args.repo:
            print("错误: --update 需要指定 --repo 参数")
            return 1
        try:
            repo_root, method = get_repo_root(args.repo, args.repo_root, 
                                              update_to_master=True)
            print(f"仓库已更新到 master 分支最新代码")
            print(f"仓库路径: {repo_root}")
        except ValueError as e:
            print(f"错误: {e}")
            return 1
        return 0
    
    if args.reports_dir:
        try:
            if args.repo:
                repo_root, repo_method = get_repo_root(args.repo, args.repo_root, update_to_master=False)
                reports_dir, reports_method = get_reports_output_dir(repo_type=args.repo, repo_root_arg=args.repo_root)
                print(f"Reports 输出目录: {reports_dir}")
                print(f"检测方式: {reports_method}")
            else:
                reports_dir, method = get_reports_output_dir(repo_type=None)
                print(f"Reports 输出目录: {reports_dir}")
                print(f"检测方式: {method}")
        except ValueError as e:
            print(f"错误: {e}")
            print("提示: 使用 --repo-root 参数指定仓库路径")
            return 1
        return 0
    
    try:
        if args.repo:
            repo_root, method = get_repo_root(args.repo, args.repo_root)
            print(f"仓库类型: {args.repo}")
            print(f"仓库路径: {repo_root}")
            print(f"检测方法: {method}")
            
            reports_dir, reports_method = get_reports_output_dir(repo_type=args.repo)
            print(f"Reports 目录: {reports_dir}")
            print(f"Reports 检测方式: {reports_method}")
        else:
            detected_name, detected_path = detect_ops_repo()
            if detected_path:
                print(f"自动检测到仓库: {detected_name}")
                print(f"仓库路径: {detected_path}")
                
                reports_dir, reports_method = get_reports_output_dir(repo_type=detected_name)
                print(f"Reports 目录: {reports_dir}")
                print(f"Reports 检测方式: {reports_method}")
            else:
                print("未检测到任何 ops-* 仓库")
                print("请在以下位置之一运行命令:")
                print("  - 包含 ops-* 子目录的目录")
                print("  - ops-* 仓库目录或其子目录")
                print("  - 或使用 --clone --repo ops-math 自动克隆")
    except ValueError as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())