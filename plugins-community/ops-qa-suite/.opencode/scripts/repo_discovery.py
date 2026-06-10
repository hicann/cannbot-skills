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
仓库动态发现模块

功能：
1. 动态发现 ops-* 仓库（无需硬编码仓库列表）
2. 动态发现分类目录（扫描算子目录结构）
3. 从 op_list.md 解析标记符号
4. 推断 GitCode 克隆 URL

核心原则：
- 约定优于配置：通过目录结构推断仓库特性
- 最小假设：只依赖标准 Ascend C 算子结构
- 自动发现：无需手动维护仓库列表

用法：
    from repo_discovery import RepoDiscovery
    
    # 发现所有仓库
    repos = RepoDiscovery.discover_repos(workspace)
    
    # 获取仓库画像
    profile = RepoDiscovery.get_repo_profile(repo_root)
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from config_loader import (
        get_repo_pattern, get_repo_markers, get_min_markers,
        get_excluded_dirs, get_structure_dirs, get_ut_structure,
        get_architecture_dirs, get_op_list_path, get_op_api_list_path,
        build_clone_url
    )
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False


class RepoDiscovery:
    """仓库动态发现器"""
    
    if USE_CONFIG:
        REPO_MARKERS = get_repo_markers()
        MIN_MARKERS_REQUIRED = get_min_markers()
        OPERATOR_MARKERS = ['op_host', 'op_kernel', 'op_api', 'CMakeLists.txt']
        GITCODE_URL_TEMPLATE = build_clone_url('{repo}').replace('{repo}', '{repo}')
        STANDARD_STRUCTURE = get_structure_dirs()
        STANDARD_UT_STRUCTURE = get_ut_structure()
        STANDARD_DOCS = {
            'op_list': get_op_list_path(),
            'op_api_list': get_op_api_list_path(),
            'readme': 'README.md',
        }
    else:
        REPO_MARKERS = ['CMakeLists.txt', 'docs', 'op_host', 'op_kernel', 'common', 'cmake']
        MIN_MARKERS_REQUIRED = 2
        OPERATOR_MARKERS = ['op_host', 'op_kernel', 'op_api', 'CMakeLists.txt']
        GITCODE_URL_TEMPLATE = "https://gitcode.com/cann/{repo}.git"
        STANDARD_STRUCTURE = {
            'host': 'op_host',
            'kernel': 'op_kernel',
            'kernel_aicpu': 'op_kernel_aicpu',
            'api': 'op_api',
            'graph': 'op_graph',
            'docs': 'docs',
        }
        STANDARD_UT_STRUCTURE = {
            'host_ut': 'tests/ut/op_host',
            'kernel_ut': 'tests/ut/op_kernel',
            'api_ut': 'tests/ut/op_api',
            'aicpu_ut': 'tests/ut/aicpu_op_kernel',
        }
        STANDARD_DOCS = {
            'op_list': 'docs/zh/op_list.md',
            'op_api_list': 'docs/zh/op_api_list.md',
            'readme': 'README.md',
        }
    
    @classmethod
    def discover_repos(cls, workspace: Path, pattern: str = "ops-") -> List[Tuple[str, Path]]:
        """
        动态发现 ops-* 仓库
        
        Args:
            workspace: 工作空间目录
            pattern: 仓库名匹配模式（默认 ops-，使用 startswith 匹配）
        
        Returns:
            List[Tuple[str, Path]]: [(repo_name, repo_path), ...]
        """
        if not workspace.exists():
            return []
        
        repos = []
        
        for item in workspace.iterdir():
            if not item.is_dir():
                continue
            
            repo_pattern = pattern if pattern else get_repo_pattern() if USE_CONFIG else "ops-"
            if not item.name.startswith(repo_pattern):
                continue
            
            if cls._is_valid_repo(item):
                repos.append((item.name, item))
        
        return repos
    
    @classmethod
    def discover_categories(cls, repo_root: Path) -> List[str]:
        """
        动态发现分类目录
        
        分类目录是包含算子的顶级目录，识别规则：
        1. 目录下存在算子子目录（包含 op_host/op_kernel 等特征）
        2. 排除特殊目录（tests, docs, common, cmake, examples 等）
        
        Args:
            repo_root: 仓库根目录
        
        Returns:
            List[str]: 分类目录列表（按名称排序）
        """
        repo_root = Path(repo_root)
        if not repo_root.exists():
            return []
        
        categories = []
        excluded_dirs = get_excluded_dirs() if USE_CONFIG else {
            'tests', 'docs', 'common', 'cmake', 'examples',
            'op_host', 'op_kernel', 'op_api', 'op_graph',
            '.git', '.github', '.opencode', 'build', 'framework'
        }
        
        for item in repo_root.iterdir():
            if not item.is_dir():
                continue
            
            if item.name in excluded_dirs or item.name.startswith('.'):
                continue
            
            if cls._is_category_dir(item):
                categories.append(item.name)
        
        return sorted(categories)
    
    @classmethod
    def discover_operators(cls, repo_root: Path, category: str = None) -> List[Tuple[str, Path]]:
        """
        发现仓库中的所有算子
        
        Args:
            repo_root: 仓库根目录
            category: 可选，限定分类目录
        
        Returns:
            List[Tuple[str, Path]]: [(op_name, op_path), ...]
        """
        repo_root = Path(repo_root)
        operators = []
        
        if category:
            category_path = repo_root / category
            if category_path.exists():
                operators.extend(cls._find_operators_in_category(category_path))
        else:
            for cat in cls.discover_categories(repo_root):
                category_path = repo_root / cat
                operators.extend(cls._find_operators_in_category(category_path))
        
        return sorted(operators, key=lambda x: x[0])
    
    @classmethod
    def discover_marker_symbols(cls, repo_root: Path) -> Dict[str, str]:
        """
        从 op_list.md 解析标记符号
        
        解析 docs/zh/op_list.md 中实际使用的实现状态标记符号
        """
        op_list_path = Path(repo_root) / cls.STANDARD_DOCS['op_list']
        
        if not op_list_path.exists():
            return {'implemented': '✓', 'not_implemented': '✗'}  # 默认
        
        try:
            content = op_list_path.read_text(encoding='utf-8')
            
            # 从 tbody 中提取标记符号
            tbody_match = re.search(r'<tbody>(.*?)</tbody>', content, re.DOTALL)
            if not tbody_match:
                return {'implemented': '✓', 'not_implemented': '✗'}
            
            tbody = tbody_match.group(1)
            
            # 查找所有 td 内容中的标记符号
            # 常见符号：√、×、✓、✗、&check;、&cross;
            markers_found = []
            
            td_pattern = r'<td[^>]*>(.*?)</td>'
            for td_match in re.finditer(td_pattern, tbody, re.DOTALL):
                td_content = td_match.group(1).strip()
                
                # 提取符号（去除 HTML 标签后的纯文本）
                clean_content = re.sub(r'<[^>]+>', '', td_content).strip()
                
                if clean_content in ['√', '×', '✓', '✗', '✅', '❌', '&check;', '&cross;', '']:
                    markers_found.append(clean_content)
            
            # 按已知符号集合分类，而非依赖出现顺序
            if markers_found:
                implemented_symbols = {'✓', '✅', '√', '✔', '√', '&check;', 'Y', 'y'}
                not_implemented_symbols = {'✗', '×', '❌', '✕', '&cross;', 'N', 'n', '-'}
                
                non_empty = [m for m in markers_found if m]
                implemented = None
                not_implemented = None
                
                for sym in non_empty:
                    if sym in implemented_symbols and implemented is None:
                        implemented = sym
                    elif sym in not_implemented_symbols and not_implemented is None:
                        not_implemented = sym
                
                if implemented and not_implemented:
                    return {'implemented': implemented, 'not_implemented': not_implemented}
                elif implemented:
                    return {'implemented': implemented, 'not_implemented': '✗'}
                elif not_implemented:
                    return {'implemented': '✓', 'not_implemented': not_implemented}
            
            return {'implemented': '✓', 'not_implemented': '✗'}
            
        except Exception:
            return {'implemented': '✓', 'not_implemented': '✗'}
    
    @classmethod
    def infer_gitcode_url(cls, repo_name: str) -> str:
        """从仓库名推断 GitCode 克隆 URL"""
        return cls.GITCODE_URL_TEMPLATE.format(repo=repo_name)
    
    @classmethod
    def get_repo_profile(cls, repo_root: Path) -> Dict:
        """
        获取完整仓库画像
        
        Returns:
            Dict: {
                'name': 仓库名,
                'path': 仓库路径,
                'categories': 分类列表,
                'markers': 标记符号,
                'gitcode_url': 克隆URL,
                'operator_count': 算子数量,
                'has_op_list': 是否有op_list.md,
                'has_op_api_list': 是否有op_api_list.md,
            }
        """
        repo_name = repo_root.name
        
        categories = cls.discover_categories(repo_root)
        markers = cls.discover_marker_symbols(repo_root)
        operators = cls.discover_operators(repo_root)
        
        return {
            'name': repo_name,
            'path': str(repo_root),
            'categories': categories,
            'markers': markers,
            'gitcode_url': cls.infer_gitcode_url(repo_name),
            'operator_count': len(operators),
            'has_op_list': (repo_root / cls.STANDARD_DOCS['op_list']).exists(),
            'has_op_api_list': (repo_root / cls.STANDARD_DOCS['op_api_list']).exists(),
            'structure': cls.STANDARD_STRUCTURE,
            'ut_structure': cls.STANDARD_UT_STRUCTURE,
            'docs_paths': cls.STANDARD_DOCS,
        }
    
    @classmethod
    def discover_common_modules(cls, repo_root: Path) -> Dict[str, Path]:
        """
        发现公共模块目录
        
        公共模块特征：
        - 目录名包含 'common', 'utils', 'shared'
        - 位于分类目录下
        - 包含 op_host 目录
        """
        common_modules = {}
        
        for category in cls.discover_categories(repo_root):
            category_path = repo_root / category
            
            for item in category_path.iterdir():
                if not item.is_dir():
                    continue
                
                # 检查是否是公共模块目录
                if any(keyword in item.name.lower() for keyword in ['common', 'utils', 'shared']):
                    if (item / 'op_host').exists():
                        common_modules[f"{category}/{item.name}"] = item
        
        return common_modules
    
    @classmethod
    def get_category_for_operator(cls, op_path: Path, repo_root: Path) -> Optional[str]:
        """根据算子路径推断其所属分类"""
        try:
            rel_path = op_path.relative_to(repo_root)
            parts = rel_path.parts
            if len(parts) >= 1:
                return parts[0]
        except ValueError:
            pass
        return None
    
    @classmethod
    def _is_valid_repo(cls, repo_path: Path) -> bool:
        """验证目录是否是有效的 ops-* 仓库"""
        if not repo_path.exists() or not repo_path.is_dir():
            return False
        
        markers_found = sum(1 for marker in cls.REPO_MARKERS 
                           if (repo_path / marker).exists())
        
        return markers_found >= cls.MIN_MARKERS_REQUIRED
    
    @classmethod
    def _is_category_dir(cls, category_path: Path) -> bool:
        """判断是否是分类目录（包含算子）"""
        for subitem in category_path.iterdir():
            if not subitem.is_dir():
                continue
            
            for marker in cls.OPERATOR_MARKERS:
                if (subitem / marker).exists():
                    return True
        
        return False
    
    @classmethod
    def _find_operators_in_category(cls, category_path: Path) -> List[Tuple[str, Path]]:
        """在分类目录中查找算子"""
        operators = []
        
        for item in category_path.iterdir():
            if not item.is_dir():
                continue
            
            if (item / 'op_host').exists() or (item / 'op_kernel').exists():
                operators.append((item.name, item))
        
        return operators


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='仓库动态发现工具')
    parser.add_argument('--workspace', type=str, default='.',
                        help='工作空间目录')
    parser.add_argument('--repo', type=str, default=None,
                        help='指定仓库路径')
    parser.add_argument('--profile', action='store_true',
                        help='显示仓库画像')
    parser.add_argument('--categories', action='store_true',
                        help='显示分类目录')
    parser.add_argument('--operators', action='store_true',
                        help='显示算子列表')
    parser.add_argument('--markers', action='store_true',
                        help='显示标记符号')
    
    args = parser.parse_args()
    
    workspace = Path(args.workspace)
    
    if args.repo:
        repo_root = Path(args.repo)
        if not repo_root.exists():
            print(f"仓库路径不存在: {repo_root}")
            return 1
        
        if args.profile:
            profile = RepoDiscovery.get_repo_profile(repo_root)
            print(f"=== {profile['name']} 仓库画像 ===")
            print(f"路径: {profile['path']}")
            print(f"分类目录: {profile['categories']}")
            print(f"标记符号: {profile['markers']}")
            print(f"克隆URL: {profile['gitcode_url']}")
            print(f"算子数量: {profile['operator_count']}")
            print(f"有op_list.md: {profile['has_op_list']}")
            print(f"有op_api_list.md: {profile['has_op_api_list']}")
            return 0
        
        if args.categories:
            categories = RepoDiscovery.discover_categories(repo_root)
            print(f"分类目录 ({len(categories)} 个):")
            for cat in categories:
                ops = RepoDiscovery.discover_operators(repo_root, cat)
                print(f"  - {cat}: {len(ops)} 个算子")
            return 0
        
        if args.operators:
            operators = RepoDiscovery.discover_operators(repo_root)
            print(f"算子列表 ({len(operators)} 个):")
            for op_name, _ in operators[:20]:
                print(f"  - {op_name}")
            if len(operators) > 20:
                print(f"  ... 共 {len(operators)} 个")
            return 0
        
        if args.markers:
            markers = RepoDiscovery.discover_marker_symbols(repo_root)
            print(f"标记符号:")
            print(f"  已实现: {markers['implemented']}")
            print(f"  未实现: {markers['not_implemented']}")
            return 0
        
        # 默认显示基本信息
        categories = RepoDiscovery.discover_categories(repo_root)
        markers = RepoDiscovery.discover_marker_symbols(repo_root)
        print(f"仓库: {repo_root.name}")
        print(f"分类数: {len(categories)}")
        print(f"标记: {markers}")
        return 0
    
    # 发现所有仓库
    repos = RepoDiscovery.discover_repos(workspace)
    
    if not repos:
        print(f"未找到 ops-* 仓库")
        return 1
    
    print(f"发现 {len(repos)} 个仓库:")
    for repo_name, repo_path in repos:
        categories = RepoDiscovery.discover_categories(repo_path)
        print(f"  - {repo_name}: {len(categories)} 个分类")
    
    return 0


if __name__ == '__main__':
    exit(main())