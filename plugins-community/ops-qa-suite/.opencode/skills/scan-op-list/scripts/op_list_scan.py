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
算子列表一致性扫描脚本（动态发现版本）

用于验证仓库级算子列表文档（docs/zh/op_list.md）与实际算子实现的一致性：
- 算子目录是否存在且路径正确
- 算子分类是否与实际目录结构一致
- 实现状态标记（√×）是否与实际文件存在一致
- 硬件单元说明是否与实际实现一致
- 文档链接是否可正常跳转

核心改进：
- 动态发现分类目录（无需硬编码）
- 自动解析标记符号（从 op_list.md 提取）
- 支持任意 ops-* 仓库

用法:
    python op_list_scan.py --repo ops-nn
    python op_list_scan.py --repo ops-math
"""

import argparse
import os
import sys
import re
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from html.parser import HTMLParser
from collections import defaultdict

logger = logging.getLogger(__name__)

_scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scripts')
sys.path.insert(0, os.path.abspath(_scripts_dir))
from repo_detector import get_repo_root, get_reports_output_dir
from repo_discovery import RepoDiscovery
from config_loader import get_structure_dirs, get_op_list_path

STRUCTURE = get_structure_dirs()
OP_LIST_DOC = get_op_list_path()


class OpListHTMLParser(HTMLParser):
    """解析 op_list.md 中的 HTML 表格"""
    
    def __init__(self):
        super().__init__()
        self.operators = []
        self.current_row = []
        self.current_cell = ''
        self.in_table = False
        self.in_tbody = False
        self.in_row = False
        self.in_cell = False
        self.row_count = 0
    
    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.in_table = True
        elif tag == 'tbody' and self.in_table:
            self.in_tbody = True
        elif tag == 'tr' and self.in_tbody:
            self.in_row = True
            self.current_row = []
            self.row_count += 1
        elif tag == 'td' and self.in_row:
            self.in_cell = True
            self.current_cell = ''
    
    def handle_endtag(self, tag):
        if tag == 'table':
            self.in_table = False
            self.in_tbody = False
        elif tag == 'tbody':
            self.in_tbody = False
        elif tag == 'tr' and self.in_row:
            self.in_row = False
            self.in_cell = False
            if len(self.current_row) >= 7:
                self._parse_operator_row()
        elif tag == 'td' and self.in_cell:
            self.in_cell = False
            self.current_row.append(self.current_cell.strip())
    
    def handle_data(self, data):
        if self.in_cell:
            self.current_cell += data
    
    def handle_entityref(self, name):
        if self.in_cell:
            if name == 'check':
                self.current_cell += '✓'
            elif name == 'cross':
                self.current_cell += '✗'
            else:
                self.current_cell += f'&{name};'
    
    def _parse_operator_row(self):
        """解析算子行数据"""
        if len(self.current_row) < 7:
            return
        
        category = self.current_row[0]
        op_dir_cell = self.current_row[1]
        
        op_name = None
        op_path = None
        link_path = ''
        
        link_match = re.search(r'<a href="([^"]+)">([^<]+)</a>', op_dir_cell)
        if link_match:
            link_path = link_match.group(1)
            op_name = link_match.group(2)
            path_match = re.search(r'\.\./([^/]+)/([^/]+)/README\.md', link_path)
            if path_match:
                op_path = f"{path_match.group(1)}/{path_match.group(2)}"
        
        op_kernel = self.current_row[2]
        op_host = self.current_row[3]
        op_api = self.current_row[4]
        op_graph = self.current_row[5]
        hardware = self.current_row[6] if len(self.current_row) > 6 else ''
        description = self.current_row[7] if len(self.current_row) > 7 else ''
        
        if op_name and op_path:
            self.operators.append({
                'category': category,
                'name': op_name,
                'path': op_path,
                'link_path': link_path,
                'op_kernel': op_kernel,
                'op_host': op_host,
                'op_api': op_api,
                'op_graph': op_graph,
                'hardware': hardware,
                'description': description,
            })


def parse_op_list_file(op_list_file):
    """解析 op_list.md 文件（使用正则表达式）"""
    with open(op_list_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取 tbody 内容
    tbody_match = re.search(r'<tbody>(.*?)</tbody>', content, re.DOTALL)
    if not tbody_match:
        logger.warning("未找到 tbody 标签")
        return []
    
    tbody_content = tbody_match.group(1)
    
    # 提取所有 tr 行
    tr_pattern = r'<tr>(.*?)</tr>'
    tr_matches = re.findall(tr_pattern, tbody_content, re.DOTALL)
    
    operators = []
    
    for tr_content in tr_matches:
        # 提取所有 td 单元格
        td_pattern = r'<td>(.*?)</td>'
        td_matches = re.findall(td_pattern, tr_content, re.DOTALL)
        
        if len(td_matches) < 7:
            continue
        
        category = td_matches[0].strip()
        op_dir_cell = td_matches[1].strip()
        
        # 从链接中提取算子名和路径
        link_match = re.search(r'<a href="([^"]+)">([^<]+)</a>', op_dir_cell)
        if not link_match:
            continue
        
        link_path = link_match.group(1)
        op_name = link_match.group(2)
        
        # 提取路径
        path_match = re.search(r'\.\./([^/]+)/([^/]+)/README\.md', link_path)
        if not path_match:
            continue
        
        op_path = f"{path_match.group(1)}/{path_match.group(2)}"
        
        # 提取标记（处理不同的符号）
        op_kernel = td_matches[2].strip()
        op_host = td_matches[3].strip()
        op_api = td_matches[4].strip()
        op_graph = td_matches[5].strip()
        hardware = td_matches[6].strip() if len(td_matches) > 6 else ''
        description = td_matches[7].strip() if len(td_matches) > 7 else ''
        
        operators.append({
            'category': category,
            'name': op_name,
            'path': op_path,
            'link_path': link_path,
            'op_kernel': op_kernel,
            'op_host': op_host,
            'op_api': op_api,
            'op_graph': op_graph,
            'hardware': hardware,
            'description': description,
        })
    
    return operators


def get_category_dirs(repo_root):
    """动态发现仓库中的分类目录
    
    分类目录特征：
    - 位于仓库根目录下
    - 包含算子子目录（非公共模块）
    - 不包括 experimental 目录
    
    Args:
        repo_root: 仓库根目录
    
    Returns:
        list: 分类目录列表
    """
    category_dirs = []
    skip_dirs = ['experimental', 'docs', 'tests', 'examples', 'scripts', 'tools', '.git']
    
    for item in os.listdir(repo_root):
        item_path = os.path.join(repo_root, item)
        
        if not os.path.isdir(item_path):
            continue
        
        if item in skip_dirs or item.startswith('.') or item.startswith('_'):
            continue
        
        if item.lower() in ['cmake', 'build', 'reports']:
            continue
        
        category_dirs.append(item)
    
    return category_dirs


def is_real_operator(op_dir):
    """判断目录是否为真正的算子目录
    
    判断规则：
    - 有 op_kernel/*.cpp 或 op_kernel_aicpu/*.cpp → 算子
    - 有 op_api/aclnn_*.cpp 或 ACLNNTYPE=aclnn/aclnn_inner → 算子
    - 其他情况 → 非算子（公共模块等）
    
    Args:
        op_dir: 算子目录路径
    
    Returns:
        bool: 是否为真正的算子目录
    """
    has_kernel, _ = check_op_kernel(op_dir)
    has_l2, _, aclnn_type, _, _ = check_op_api(op_dir)
    
    if has_kernel:
        return True
    
    if has_l2:
        return True
    
    if aclnn_type in ['aclnn', 'aclnn_inner']:
        return True
    
    return False


def find_missing_operators(repo_root, listed_op_paths):
    """排查算子遗漏
    
    遍历所有分类目录下的子目录，检查是否有算子遗漏在 op_list 中。
    
    Args:
        repo_root: 仓库根目录
        listed_op_paths: op_list 中已列出的算子路径集合
    
    Returns:
        list: 遗漏的算子列表 [{category, name, path, reason}]
    """
    missing = []
    
    category_dirs = get_category_dirs(repo_root)
    
    for category in category_dirs:
        category_path = os.path.join(repo_root, category)
        
        if not os.path.isdir(category_path):
            continue
        
        for item in os.listdir(category_path):
            item_path = os.path.join(category_path, item)
            
            if not os.path.isdir(item_path):
                continue
            
            op_path = f"{category}/{item}"
            
            if is_experimental_op(op_path):
                continue
            
            if op_path in listed_op_paths:
                continue
            
            if is_real_operator(item_path):
                missing.append({
                    'category': category,
                    'name': item,
                    'path': op_path,
                    'reason': '有实现但未列入op_list'
                })
    
    return missing


def check_op_kernel(op_dir):
    op_kernel_dir = os.path.join(op_dir, STRUCTURE['kernel'])
    op_kernel_aicpu_dir = os.path.join(op_dir, STRUCTURE['kernel_aicpu'])
    
    has_aicore = False
    has_aicpu = False
    
    if os.path.exists(op_kernel_dir):
        for _, _, files in os.walk(op_kernel_dir):
            for f in files:
                if (f.endswith('.asc') or f.endswith('.cpp')) and '_def.cpp' not in f:
                    has_aicore = True
                    break
    
    if os.path.exists(op_kernel_aicpu_dir):
        for _, _, files in os.walk(op_kernel_aicpu_dir):
            for f in files:
                if f.endswith('.cpp') and '_def.cpp' not in f:
                    has_aicpu = True
                    break
    
    return has_aicore, has_aicpu


def check_op_host(op_dir, repo_root=None, op_name=None, category=None):
    op_host_dir = os.path.join(op_dir, STRUCTURE['host'])
    
    if os.path.exists(op_host_dir):
        for _, _, files in os.walk(op_host_dir):
            for f in files:
                if '_tiling' in f and f.endswith('.cpp'):
                    return True
                if '_infershape' in f and f.endswith('.cpp'):
                    return True
    
    if repo_root and op_name and category:
        common_module_path = get_common_module_path(repo_root, category, op_name)
        if common_module_path:
            if check_foreach_common_impl(common_module_path, op_name):
                return True
    
    return False


def check_foreach_common_impl(foreach_utils_host, op_name):
    """检查公共模块中是否有算子的 tiling/infershape 注册
    
    Args:
        foreach_utils_host: 公共模块 op_host 目录路径
        op_name: 算子名称
    
    Returns:
        bool: 是否找到注册
    """
    op_class_name = ''.join(word.capitalize() for word in op_name.split('_'))
    
    files_to_check = [
        'foreach_tiling_func.cpp',
        'foreach_infershape.cpp',
        'foreach_regbase_tiling.cpp',
        'foreach_reduce_regbase_tiling.cpp',
        'foreach_reduce_tiling_func.cpp',
        'conv_forward_infershape.cpp',
        'conv_backprop_infershape.cpp',
        'matmul_common_infershape.cpp',
    ]
    
    for filename in files_to_check:
        filepath = os.path.join(foreach_utils_host, filename)
        if not os.path.exists(filepath):
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        patterns = [
            f'IMPL_OP_OPTILING\\({op_class_name}\\)',
            f'IMPL_OP_INFERSHAPE\\({op_class_name}\\)',
            f'REGISTER_OPS_TILING_TEMPLATE\\({op_class_name}',
            f'Tiling4{op_class_name}Tiling',
            f'TilingFunc4{op_class_name}',
        ]
        
        for pattern in patterns:
            if re.search(pattern, content):
                return True
    
    return False


def get_common_module_path(repo_root, category, op_name):
    """根据分类动态发现公共模块路径
    
    公共模块特征：
    - 目录名包含 'common', 'utils', 'shared', '_utils', '_common'
    - 位于分类目录下
    - 包含 op_host 目录
    
    Args:
        repo_root: 仓库根目录
        category: 算子分类（如 foreach, conv, matmul, norm 等）
        op_name: 算子名称
    
    Returns:
        str: 公共模块 op_host 目录路径，或 None
    """
    category_dir = os.path.join(repo_root, category)
    
    if not os.path.exists(category_dir):
        return None
    
    common_keywords = ['common', 'utils', 'shared', '_utils', '_common']
    
    for item in os.listdir(category_dir):
        item_path = os.path.join(category_dir, item)
        
        if not os.path.isdir(item_path):
            continue
        
        if any(kw in item.lower() for kw in common_keywords):
            op_host_path = os.path.join(item_path, STRUCTURE['host'])
            if os.path.exists(op_host_path):
                return op_host_path
    
    return None


def check_op_api(op_dir):
    op_api_dir = os.path.join(op_dir, STRUCTURE['api'])
    op_host_api_dir = os.path.join(op_dir, STRUCTURE['host'], STRUCTURE['api'])
    cmake_file = os.path.join(op_dir, 'CMakeLists.txt')
    host_cmake_file = os.path.join(op_dir, STRUCTURE['host'], 'CMakeLists.txt')
    
    aclnn_type = None
    for cmake in [cmake_file, host_cmake_file]:
        if os.path.exists(cmake):
            with open(cmake, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.search(r'ACLNNTYPE\s+(\w+)', content)
                if match:
                    aclnn_type = match.group(1)
                    break
    
    has_l2 = False
    has_l0 = False
    has_nested_api = False
    
    if os.path.exists(op_api_dir):
        for f in os.listdir(op_api_dir):
            if f.endswith('.cpp') or f.endswith('.h'):
                if f.startswith('aclnn_'):
                    has_l2 = True
                elif f not in '_def':
                    has_l0 = True
    
    if os.path.exists(op_host_api_dir):
        has_nested_api = True
        for _, _, files in os.walk(op_host_api_dir):
            for f in files:
                if f.endswith('.cpp') or f.endswith('.h'):
                    if f.startswith('aclnn_'):
                        has_l2 = True
                    elif '_def' not in f:
                        has_l0 = True
    
    # ACLNNTYPE判断规则：aclnn=✔, aclnn_exclude+有文件=✔, aclnn_inner=不暴露
    if aclnn_type == 'aclnn':
        has_l2 = True
    elif aclnn_type == 'aclnn_exclude' and has_l2:
        pass
    
    api_status = 'none'
    if has_l2:
        api_status = 'l2'
    elif has_l0:
        api_status = 'l0_only'
    
    return has_l2, has_l0, aclnn_type, api_status, has_nested_api


def check_op_graph(op_dir):
    op_graph_dir = os.path.join(op_dir, STRUCTURE['graph'])
    
    if not os.path.exists(op_graph_dir):
        return False
    
    for f in os.listdir(op_graph_dir):
        if '_proto' in f:
            return True
    
    return False


def check_launcher_macro(op_dir, op_name):
    """检查L0文件中的LAUNCHER宏（精确匹配算子名）
    
    Args:
        op_dir: 算子目录路径
        op_name: 算子名称（snake_case）
    
    Returns:
        tuple: (has_aicore_launcher, has_aicpu_launcher)
    """
    op_class_name = ''.join(word.capitalize() for word in op_name.split('_'))
    
    op_api_dir = os.path.join(op_dir, STRUCTURE['api'])
    host_op_api_dir = os.path.join(op_dir, STRUCTURE['host'], STRUCTURE['api'])
    
    cpp_files = []
    
    if os.path.exists(op_api_dir):
        for f in os.listdir(op_api_dir):
            if f.endswith('.cpp'):
                cpp_files.append(os.path.join(op_api_dir, f))
    
    if os.path.exists(host_op_api_dir):
        for root, _, files in os.walk(host_op_api_dir):
            for f in files:
                if f.endswith('.cpp'):
                    cpp_files.append(os.path.join(root, f))
    
    has_aicore_launcher = False
    has_aicpu_launcher = False
    
    for cpp_file in cpp_files:
        try:
            with open(cpp_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if re.search(f'ADD_TO_LAUNCHER_LIST_AICORE\\({op_class_name}', content):
                has_aicore_launcher = True
            if re.search(f'ADD_TO_LAUNCHER_LIST_AICPU\\({op_class_name}', content):
                has_aicpu_launcher = True
        except (UnicodeDecodeError, OSError):
            pass
    
    return has_aicore_launcher, has_aicpu_launcher


def get_hardware_type(op_dir, op_name=None):
    """获取硬件单元类型
    
    判断规则：
    - op_kernel/*.cpp → AI Core
    - op_kernel_aicpu/*.cpp → AI CPU
    - L0中ADD_TO_LAUNCHER_LIST_AICORE({本算子名}) → AI Core
    - L0中ADD_TO_LAUNCHER_LIST_AICPU({本算子名}) → AI CPU
    
    Args:
        op_dir: 算子目录路径
        op_name: 算子名称（用于LAUNCHER宏精确匹配）
    
    Returns:
        str: 硬件单元类型
    """
    has_aicore_kernel, has_aicpu_kernel = check_op_kernel(op_dir)
    
    has_aicore_launcher = False
    has_aicpu_launcher = False
    
    if op_name:
        has_aicore_launcher, has_aicpu_launcher = check_launcher_macro(op_dir, op_name)
    
    has_aicore = has_aicore_kernel or has_aicore_launcher
    has_aicpu = has_aicpu_kernel or has_aicpu_launcher
    
    if has_aicore and has_aicpu:
        return 'AI Core/AI CPU'
    elif has_aicpu:
        return 'AI CPU'
    elif has_aicore:
        return 'AI Core'
    else:
        return '仅API'


def is_marker_implemented(marker, repo_root):
    """判断标记是否为已实现（动态解析）"""
    symbols = RepoDiscovery.discover_marker_symbols(repo_root)
    return marker == symbols['implemented']


def is_experimental_op(op_path):
    """判断是否为 experimental 目录下的算子（生态开发者提供，不检查）
    
    Args:
        op_path: 算子路径（如 "experimental/activation/relu"）
    
    Returns:
        bool: 是否为 experimental 算子
    """
    return op_path.startswith('experimental/') or '/experimental/' in op_path


def is_placeholder_readme(readme_file):
    """判断 README.md 是否为占位文档
    
    占位文档特征：包含特定关键词表示算子暂无实现
    
    Args:
        readme_file: README.md 文件路径
    
    Returns:
        tuple: (bool, str) - 是否为占位文档, 匹配到的关键词
    """
    if not os.path.exists(readme_file):
        return False, None
    
    try:
        with open(readme_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        placeholder_keywords = [
            '该算子暂无Ascend C代码实现',
            '欢迎开发者补充贡献',
            '暂无实现',
            '待实现',
            '待开发',
        ]
        
        for keyword in placeholder_keywords:
            if keyword in content:
                return True, keyword
        
        return False, None
    except Exception:
        return False, None


def scan_repository(repo_root, repo_type):
    op_list_file = os.path.join(repo_root, OP_LIST_DOC)
    
    if not os.path.exists(op_list_file):
        return {'error': f'op_list.md 文件不存在: {op_list_file}'}
    
    operators = parse_op_list_file(op_list_file)
    logger.info(f"解析到 {len(operators)} 个算子")
    
    results = []
    issues = {
        'directory_missing': [],
        'readme_missing': [],
        'readme_placeholder': [],
        'category_error': [],
        'kernel_marker_error': [],
        'host_marker_error': [],
        'api_marker_error': [],
        'graph_marker_error': [],
        'hardware_error': [],
        'link_error': [],
        'experimental_skipped': [],
        'api_structure_issue': [],
        'missing_in_doc': [],
    }
    
    stats = {
        'total': len(operators),
        'passed': 0,
        'failed': 0,
        'by_check': {
            'directory': {'total': 0, 'passed': 0},
            'category': {'total': 0, 'passed': 0},
            'kernel_marker': {'total': 0, 'passed': 0},
            'host_marker': {'total': 0, 'passed': 0},
            'api_marker': {'total': 0, 'passed': 0},
            'graph_marker': {'total': 0, 'passed': 0},
            'hardware': {'total': 0, 'passed': 0},
        }
    }
    
    for op in operators:
        op_dir = os.path.join(repo_root, op['path'])
        result = {
            'name': op['name'],
            'category': op['category'],
            'path': op['path'],
            'checks': {},
            'status': 'passed',
        }
        
        # 跳过 experimental 目录下的算子（生态开发者提供，不检查）
        if is_experimental_op(op['path']):
            result['checks']['directory'] = 'skipped_experimental'
            result['status'] = 'skipped'
            stats['total'] -= 1
            issues['experimental_skipped'].append(op)
            results.append(result)
            continue
        
        # 检查1：目录存在性
        stats['by_check']['directory']['total'] += 1
        if not os.path.exists(op_dir):
            result['checks']['directory'] = 'missing'
            result['status'] = 'failed'
            issues['directory_missing'].append(op)
            stats['by_check']['directory']['passed'] += 0
        else:
            # 检查 README.md（必选交付件）
            readme_file = os.path.join(op_dir, 'README.md')
            if not os.path.exists(readme_file):
                result['checks']['directory'] = 'readme_missing'
                result['status'] = 'failed'
                issues['readme_missing'].append(op)
            else:
                is_placeholder, placeholder_keyword = is_placeholder_readme(readme_file)
                if is_placeholder:
                    result['checks']['readme'] = f'placeholder: {placeholder_keyword}'
                    issues['readme_placeholder'].append({**op, 'keyword': placeholder_keyword})
                    # 占位文档不标记为失败，仅标注
                else:
                    result['checks']['directory'] = 'passed'
                    stats['by_check']['directory']['passed'] += 1
        
        # 检查2：分类正确性
        if os.path.exists(op_dir):
            stats['by_check']['category']['total'] += 1
            actual_category = os.path.basename(os.path.dirname(op_dir))
            if op['category'] != actual_category:
                result['checks']['category'] = f'error: doc={op["category"]}, actual={actual_category}'
                result['status'] = 'failed'
                issues['category_error'].append({**op, 'actual_category': actual_category})
            else:
                result['checks']['category'] = 'passed'
                stats['by_check']['category']['passed'] += 1
        
        # 检查3-6：实现标记一致性
        if os.path.exists(op_dir):
            # op_kernel
            stats['by_check']['kernel_marker']['total'] += 1
            has_aicore, has_aicpu = check_op_kernel(op_dir)
            has_kernel = has_aicore or has_aicpu
            doc_has_kernel = is_marker_implemented(op['op_kernel'], repo_root)
            
            if has_kernel != doc_has_kernel:
                result['checks']['kernel_marker'] = f'error: doc={op["op_kernel"]}, actual={has_kernel}'
                result['status'] = 'failed'
                issues['kernel_marker_error'].append({**op, 'actual': has_kernel})
            else:
                result['checks']['kernel_marker'] = 'passed'
                stats['by_check']['kernel_marker']['passed'] += 1
            
            # op_host
            stats['by_check']['host_marker']['total'] += 1
            has_host = check_op_host(op_dir, repo_root, op['name'], op['category'])
            doc_has_host = is_marker_implemented(op['op_host'], repo_root)
            
            if has_host != doc_has_host:
                result['checks']['host_marker'] = f'error: doc={op["op_host"]}, actual={has_host}'
                result['status'] = 'failed'
                issues['host_marker_error'].append({**op, 'actual': has_host})
            else:
                result['checks']['host_marker'] = 'passed'
                stats['by_check']['host_marker']['passed'] += 1
            
            # op_api
            stats['by_check']['api_marker']['total'] += 1
            has_l2, has_l0, aclnn_type, api_status, has_nested_api = check_op_api(op_dir)
            doc_has_api = is_marker_implemented(op['op_api'], repo_root)
            
            if api_status == 'l2':
                has_api = True
            elif api_status == 'l0_only':
                has_api = False
            else:
                has_api = False
            
            if has_api != doc_has_api:
                if api_status == 'l0_only':
                    result['checks']['api_marker'] = f'error: doc={op["op_api"]}, actual=仅L0接口'
                    issues['api_marker_error'].append(
                {**op, 'actual': 'l0_only', 'aclnn_type': aclnn_type,
                 'api_status': api_status}
            )
                else:
                    result['checks']['api_marker'] = f'error: doc={op["op_api"]}, actual={has_api}'
                    issues['api_marker_error'].append(
                {**op, 'actual': has_api, 'aclnn_type': aclnn_type,
                 'api_status': api_status}
            )
                result['status'] = 'failed'
            else:
                result['checks']['api_marker'] = 'passed'
                stats['by_check']['api_marker']['passed'] += 1
            
            result['api_status'] = api_status
            
            # 特殊场景：op_host/op_api 嵌套结构需整改
            if has_nested_api and has_l2:
                result['checks']['api_structure'] = 'needs_restructure'
                issues['api_structure_issue'].append({
                    **op,
                    'reason': 'op_host/op_api嵌套结构，op_api应与op_host同级'
                })
            
            # op_graph
            stats['by_check']['graph_marker']['total'] += 1
            has_graph = check_op_graph(op_dir)
            doc_has_graph = is_marker_implemented(op['op_graph'], repo_root)
            
            if has_graph != doc_has_graph:
                result['checks']['graph_marker'] = f'error: doc={op["op_graph"]}, actual={has_graph}'
                result['status'] = 'failed'
                issues['graph_marker_error'].append({**op, 'actual': has_graph})
            else:
                result['checks']['graph_marker'] = 'passed'
                stats['by_check']['graph_marker']['passed'] += 1
            
            # 硬件单元
            stats['by_check']['hardware']['total'] += 1
            actual_hardware = get_hardware_type(op_dir, op['name'])
            if op['hardware'] and actual_hardware != op['hardware']:
                result['checks']['hardware'] = f'error: doc={op["hardware"]}, actual={actual_hardware}'
                result['status'] = 'failed'
                issues['hardware_error'].append({**op, 'actual': actual_hardware})
            else:
                result['checks']['hardware'] = 'passed'
                stats['by_check']['hardware']['passed'] += 1
        
        results.append(result)
        
        if result['status'] == 'passed':
            stats['passed'] += 1
        else:
            stats['failed'] += 1
    
    # 遗漏排查：检查是否有算子未列入 op_list
    listed_op_paths = set(op['path'] for op in operators)
    missing_ops = find_missing_operators(repo_root, listed_op_paths)
    
    if missing_ops:
        issues['missing_in_doc'] = missing_ops
        stats['failed'] += len(missing_ops)
    
    return {
        'stats': stats,
        'issues': issues,
        'results': results,
        'repo_type': repo_type,
        'scan_time': datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        'missing_in_doc': missing_ops,
    }


def generate_report(scan_data, output_path):
    """生成 Markdown 报告"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    repo_type = scan_data['repo_type']
    stats = scan_data['stats']
    issues = scan_data['issues']
    results = scan_data['results']
    
    report = f"""# {repo_type} 算子列表一致性扫描报告

**扫描时间**: {scan_data['scan_time']}  
**仓库路径**: `{repo_type}/`  
**扫描范围**: docs/zh/op_list.md（全量验证）

---

## 一、扫描概览

### 1.1 统计摘要

| 检查项 | 扫描数 | 通过数 | 失败数 |
|-------|-------|-------|-------|
| 算子目录存在性 | {stats['by_check']['directory']['total']} | {stats['by_check']['directory']['passed']} | {stats['by_check']['directory']['total'] - stats['by_check']['directory']['passed']} |
| 算子分类正确性 | {stats['by_check']['category']['total']} | {stats['by_check']['category']['passed']} | {stats['by_check']['category']['total'] - stats['by_check']['category']['passed']} |
| op_kernel标记一致性 | {stats['by_check']['kernel_marker']['total']} | {stats['by_check']['kernel_marker']['passed']} | {stats['by_check']['kernel_marker']['total'] - stats['by_check']['kernel_marker']['passed']} |
| op_host标记一致性 | {stats['by_check']['host_marker']['total']} | {stats['by_check']['host_marker']['passed']} | {stats['by_check']['host_marker']['total'] - stats['by_check']['host_marker']['passed']} |
| op_api标记一致性 | {stats['by_check']['api_marker']['total']} | {stats['by_check']['api_marker']['passed']} | {stats['by_check']['api_marker']['total'] - stats['by_check']['api_marker']['passed']} |
| op_graph标记一致性 | {stats['by_check']['graph_marker']['total']} | {stats['by_check']['graph_marker']['passed']} | {stats['by_check']['graph_marker']['total'] - stats['by_check']['graph_marker']['passed']} |
| 硬件单元一致性 | {stats['by_check']['hardware']['total']} | {stats['by_check']['hardware']['passed']} | {stats['by_check']['hardware']['total'] - stats['by_check']['hardware']['passed']} |
| **总计** | **{stats['total']}** | **{stats['passed']}** | **{stats['failed']}** |

### 1.2 问题分类统计

| 问题类型 | 数量 | 占比 |
|---------|------|------|
| 目录缺失 | {len(issues['directory_missing'])} | {len(issues['directory_missing'])/max(stats['total'],1)*100:.1f}% |
| README缺失 | {len(issues['readme_missing'])} | {len(issues['readme_missing'])/max(stats['total'],1)*100:.1f}% |
| README占位文档 | {len(issues['readme_placeholder'])} | {len(issues['readme_placeholder'])/max(stats['total'],1)*100:.1f}% |
| 分类错误 | {len(issues['category_error'])} | {len(issues['category_error'])/max(stats['total'],1)*100:.1f}% |
| op_kernel标记错误 | {len(issues['kernel_marker_error'])} | {len(issues['kernel_marker_error'])/max(stats['total'],1)*100:.1f}% |
| op_host标记错误 | {len(issues['host_marker_error'])} | {len(issues['host_marker_error'])/max(stats['total'],1)*100:.1f}% |
| op_api标记错误 | {len(issues['api_marker_error'])} | {len(issues['api_marker_error'])/max(stats['total'],1)*100:.1f}% |
| op_graph标记错误 | {len(issues['graph_marker_error'])} | {len(issues['graph_marker_error'])/max(stats['total'],1)*100:.1f}% |
| 硬件单元错误 | {len(issues['hardware_error'])} | {len(issues['hardware_error'])/max(stats['total'],1)*100:.1f}% |
| 目录结构需整改 | {len(issues['api_structure_issue'])} | {len(issues['api_structure_issue'])/max(stats['total'],1)*100:.1f}% |
| 算子遗漏（未列入op_list） | {len(issues['missing_in_doc'])} | {len(issues['missing_in_doc'])/max(stats['total'],1)*100:.1f}% |
| 跳过experimental算子 | {len(issues['experimental_skipped'])} | - |

---

## 二、问题详情

"""
    
    # 目录缺失
    if issues['directory_missing']:
        report += "### 2.1 目录缺失\n\n"
        report += "| 序号 | 算子名 | 分类 | 文档路径 |\n"
        report += "|:---:|-------|------|---------|\n"
        for i, op in enumerate(issues['directory_missing'], 1):
            report += f"| {i} | {op['name']} | {op['category']} | `{op['path']}` |\n"
        report += "\n"
    
    # README缺失
    if issues['readme_missing']:
        report += "### 2.2 README缺失\n\n"
        report += "| 序号 | 算子名 | 分类 | 文档路径 |\n"
        report += "|:---:|-------|------|---------|\n"
        for i, op in enumerate(issues['readme_missing'], 1):
            report += f"| {i} | {op['name']} | {op['category']} | `{op['path']}` |\n"
        report += "\n"
    
    # README占位文档（非失败，仅标注）
    if issues['readme_placeholder']:
        report += "### 2.3 README占位文档 ✅ 非问题\n\n"
        report += "| 序号 | 算子名 | 分类 | 文档路径 | 占位关键词 |\n"
        report += "|:---:|-------|------|---------|-----------|\n"
        for i, op in enumerate(issues['readme_placeholder'], 1):
            report += f"| {i} | {op['name']} | {op['category']} | `{op['path']}` | {op.get('keyword', 'N/A')} |\n"
        report += "\n"
    
    # 标记错误
    for issue_type in ['kernel_marker_error', 'host_marker_error', 'api_marker_error', 'graph_marker_error']:
        if issues[issue_type]:
            type_name = issue_type.replace('_marker_error', '').replace('_', ' ')
            report += f"### 2.4 {type_name}标记错误\n\n"
            report += "| 序号 | 算子名 | 分类 | 文档标记 | 实际状态 |\n"
            report += "|:---:|-------|------|---------|---------|\n"
            for i, op in enumerate(issues[issue_type], 1):
                marker_field = 'op_' + issue_type.replace('_marker_error', '')
                doc_marker = op.get(marker_field, op.get('op_kernel', 'N/A'))
                actual_status = op.get('actual', 'N/A')
                if actual_status == True:
                    actual_desc = 'True（有实现）'
                elif actual_status == False:
                    actual_desc = 'False（无实现）'
                else:
                    actual_desc = str(actual_status)
                report += f"| {i} | {op['name']} | {op['category']} | `{doc_marker}` | `{actual_desc}` |\n"
            report += "\n"
    
    # 硬件单元错误
    if issues['hardware_error']:
        report += "### 2.5 硬件单元错误\n\n"
        report += "| 序号 | 算子名 | 分类 | 文档说明 | 实际硬件 |\n"
        report += "|:---:|-------|------|---------|---------|\n"
        for i, op in enumerate(issues['hardware_error'], 1):
            report += f"| {i} | {op['name']} | {op['category']} | `{op['hardware']}` | `{op.get('actual', 'N/A')}` |\n"
        report += "\n"
    
    # 目录结构需整改（op_host/op_api嵌套）
    if issues['api_structure_issue']:
        report += "### 2.6 目录结构需整改 ⚠️ op_host/op_api嵌套\n\n"
        report += "以下算子的 aclnn 实现位于 `op_host/op_api/` 目录下，应整改为 `op_api/` 与 `op_host` 同级。\n\n"
        report += "| 序号 | 算子名 | 分类 | 整改建议 |\n"
        report += "|:---:|-------|------|---------|\n"
        for i, op in enumerate(issues['api_structure_issue'], 1):
            report += f"| {i} | {op['name']} | {op['category']} | 将 op_host/op_api 移至 op_api/ 目录 |\n"
        report += "\n"
    
    # 跳过的 experimental 算子
    if issues['experimental_skipped']:
        report += "### 2.7 跳过的experimental算子 ✅ 生态开发者提供\n\n"
        report += f"共跳过 {len(issues['experimental_skipped'])} 个 experimental 目录下的算子（生态开发者提供，不检查）。\n\n"
        report += "| 序号 | 算子名 | 分类 | 文档路径 |\n"
        report += "|:---:|-------|------|---------|\n"
        for i, op in enumerate(issues['experimental_skipped'], 1):
            report += f"| {i} | {op['name']} | {op['category']} | `{op['path']}` |\n"
        report += "\n"
    
    # 遗漏的算子（未列入 op_list）
    if issues['missing_in_doc']:
        report += "### 2.8 算子遗漏 ⚠️ 未列入op_list\n\n"
        report += f"共发现 {len(issues['missing_in_doc'])} 个算子有实现但未列入 op_list.md 文档。\n\n"
        report += "| 序号 | 算子名 | 分类 | 算子路径 | 遗漏原因 |\n"
        report += "|:---:|-------|------|---------|---------|\n"
        for i, op in enumerate(issues['missing_in_doc'], 1):
            report += f"| {i} | {op['name']} | {op['category']} | `{op['path']}` | {op['reason']} |\n"
        report += "\n"
    
    report += "---\n\n**报告生成时间**: " + scan_data['scan_time'] + "\n"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"报告已保存到: {output_file}")
    return report


def main():
    parser = argparse.ArgumentParser(description='算子列表一致性扫描脚本（动态发现）')
    parser.add_argument('--repo', required=True,
                        help='仓库名（支持任意 ops-* 仓库）')
    parser.add_argument('--repo-root', default=None,
                        help='仓库根目录（默认根据 repo 类型推断）')
    parser.add_argument('--output', default=None,
                        help='输出 Markdown 报告路径')
    parser.add_argument('--json', default=None,
                        help='输出 JSON 数据路径')
    
    args = parser.parse_args()
    
    try:
        repo_root_obj, detection_method = get_repo_root(args.repo, args.repo_root)
        repo_root = str(repo_root_obj)
    except ValueError as e:
        logger.error(f"扫描失败: {e}", exc_info=True)
        return
    
    date_str = datetime.now(tz=timezone.utc).strftime('%Y%m%d')
    time_str = datetime.now(tz=timezone.utc).strftime('%H%M%S')
    
    reports_dir, reports_method = get_reports_output_dir(repo_type=args.repo)
    
    if args.output:
        output_path = args.output
    else:
        output_path = str(reports_dir / date_str / args.repo / f'op-list-validation_report_{time_str}.md')
    
    if args.json:
        json_path = args.json
    else:
        json_path = str(reports_dir / date_str / args.repo / 'op-list-validation_data.json')
    
    print(f"仓库类型: {args.repo}")
    print(f"仓库根目录: {repo_root} (检测方式: {detection_method})")
    print(f"Reports 目录: {reports_dir} (检测方式: {reports_method})")
    print(f"输出报告: {output_path}")
    print()
    
    scan_data = scan_repository(repo_root, args.repo)
    
    if 'error' in scan_data:
        print(f"错误: {scan_data['error']}")
        return
    
    # 保存 JSON 数据
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(scan_data, f, indent=2, ensure_ascii=False)
    print(f"JSON数据已保存: {json_path}")
    
    # 生成报告
    generate_report(scan_data, output_path)
    
    # 打印摘要
    stats = scan_data['stats']
    print("\n扫描摘要:")
    print(f"总算子数: {stats['total']}")
    print(f"通过: {stats['passed']}")
    print(f"失败: {stats['failed']}")
    if stats['total'] > 0:
        print(f"通过率: {stats['passed']/stats['total']*100:.1f}%")
    
    if scan_data['missing_in_doc']:
        print(f"\n实际存在但未列入op_list: {len(scan_data['missing_in_doc'])} 个")


if __name__ == '__main__':
    main()